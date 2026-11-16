// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "op/sirius_physical_fused_join_aggregate.hpp"

#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "expression_evaluator/expression_evaluator.hpp"
#include "log/logging.hpp"
#include "op/aggregate/fused_join_aggregate_impl.hpp"
#include "op/scan/owning_table_view.hpp"
#include "telemetry/nvtx.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/unary.hpp>

#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <stdexcept>
namespace sirius::op {
sirius_physical_fused_join_aggregate::sirius_physical_fused_join_aggregate(
  duckdb::vector<logical_type> outputs,
  size_t build_key,
  size_t probe_key,
  std::vector<sum_spec> sums,
  std::unique_ptr<ast::node> group,
  bool group_is_build,
  size_t cardinality)
  : sirius_physical_dense_count_join(
      {outputs[0], outputs.back()}, cardinality, build_key, probe_key, std::nullopt, 0),
    _sums(std::move(sums)),
    _group(std::move(group)),
    _group_is_build(group_is_build)
{
  type  = TYPE;
  types = std::move(outputs);
}
partition_strategy sirius_physical_fused_join_aggregate::get_partition_strategy(
  partition_sizing_input const&)
{
  if (num_gpus() != 1) throw std::runtime_error("experimental SUM join requires a single GPU");
  return {1, false, false, partition_placement::unpinned(1)};
}
std::size_t sirius_physical_fused_join_aggregate::no_history_peak_memory_estimate(
  input_stats const& stats) const
{
  return stats.bytes * 4 + 256ULL * 1024 * 1024;
}
std::string sirius_physical_fused_join_aggregate::params_to_string() const
{
  return " (probe every matching pair, sums=" + std::to_string(_sums.size()) +
         (_group ? ", grouped)" : ", scalar)");
}
std::unique_ptr<operator_data> sirius_physical_fused_join_aggregate::execute(
  operator_data const& data, ::cuda::stream_ref stream)
{
  nvtx_scoped_range nvtx_range{"sirius_physical_fused_join_aggregate::execute"};
  auto const& input = dynamic_cast<dense_count_join_input const&>(data);
  auto batches      = input.get_read_only_batches();
  if (batches.empty())
    throw std::runtime_error("experimental SUM join received no memory-space anchor");
  auto* space = batches.front().get_memory_space();
  auto mr     = space->get_default_allocator();
  std::vector<cudf::table_view> build_views, probe_views;
  for (size_t i = 0; i < batches.size(); i++) {
    auto view = sirius::get_cudf_table_view(batches[i]);
    (i < input.preserved_count() ? build_views : probe_views).push_back(view);
  }
  auto make_input = [&](std::vector<cudf::table_view> const& views, size_t side) {
    if (!views.empty()) return cudf::concatenate(views, stream, mr);
    std::vector<std::unique_ptr<cudf::column>> cols;
    for (auto const& t : children[side]->types)
      cols.push_back(cudf::make_empty_column(require_cudf_type(t)));
    return std::make_unique<cudf::table>(std::move(cols));
  };
  auto build = make_input(build_views, 0);
  auto probe = make_input(probe_views, 1);
  auto bk    = cudf::cast(
    build->view().column(preserved_key_idx()), cudf::data_type{cudf::type_id::INT64}, stream, mr);
  auto pk = cudf::cast(
    probe->view().column(counted_key_idx()), cudf::data_type{cudf::type_id::INT64}, stream, mr);
  std::vector<std::unique_ptr<cudf::table>> expression_results;
  std::vector<std::unique_ptr<cudf::column>> cast_values;
  std::vector<cudf::column_view> values, predicates;
  std::vector<bool> conditional, value_is_build, predicate_is_build;
  for (size_t k = 0; k < _sums.size(); k++) {
    expression_evaluator eval(*_sums[k].value, mr, stream, expression_policy());
    expression_results.push_back(
      eval.evaluate(_sums[k].value_is_build ? build->view() : probe->view()));
    auto dt = require_cudf_type(types[(_group ? 1 : 0) + k]);
    cast_values.push_back(cudf::cast(expression_results.back()->view().column(0), dt, stream, mr));
    values.push_back(cast_values.back()->view());
    predicates.emplace_back();
    conditional.push_back(_sums[k].conditional);
    value_is_build.push_back(_sums[k].value_is_build);
    predicate_is_build.push_back(_sums[k].predicate_is_build);
    if (_sums[k].predicate) {
      expression_evaluator pred(*_sums[k].predicate, mr, stream, expression_policy());
      expression_results.push_back(
        pred.evaluate(_sums[k].predicate_is_build ? build->view() : probe->view()));
      predicates.back() = expression_results.back()->view().column(0);
    }
  }
  std::unique_ptr<cudf::table> group_values;
  std::unique_ptr<cudf::column> encoded, ids;
  cudf::column_view keys, ids_view;
  int groups = 1;
  if (_group) {
    expression_evaluator eval(*_group, mr, stream, expression_policy());
    group_values = eval.evaluate(_group_is_build ? build->view() : probe->view());
    auto g       = group_values->view().column(0);
    if (g.has_nulls())
      throw std::runtime_error("experimental SUM join: NULL group key unsupported");
    encoded = cudf::dictionary::encode(g, cudf::data_type{cudf::type_id::INT32}, stream, mr);
    cudf::dictionary_column_view dict(encoded->view());
    keys   = dict.keys();
    groups = keys.size();
    if (groups > 4) throw std::runtime_error("experimental SUM join: group domain exceeds four");
    ids =
      cudf::cast(dict.get_indices_annotated(), cudf::data_type{cudf::type_id::INT32}, stream, mr);
    ids_view = ids->view();
  }
  std::vector<cudf::data_type> output_types;
  for (auto const& t : types)
    output_types.push_back(require_cudf_type(t));
  if (groups == 0) {
    auto out = sirius::make_empty_table(output_types);
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<::cucascade::data_batch>>{
        make_data_batch(std::move(out), *space, stream, batch_telemetry())});
  }
  auto partials = probe_sum_join(bk->view(),
                                 pk->view(),
                                 values,
                                 predicates,
                                 conditional,
                                 value_is_build,
                                 predicate_is_build,
                                 _group ? &ids_view : nullptr,
                                 _group_is_build,
                                 groups,
                                 stream,
                                 mr);
  auto sum      = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  auto part     = [&](cudf::column_view col, int group) {
    return cudf::slice(col, {group * partials.warps, (group + 1) * partials.warps}, stream)[0];
  };
  auto count = [&](cudf::column_view col, int group) {
    auto s =
      cudf::reduce(part(col, group), *sum, cudf::data_type{cudf::type_id::INT64}, stream, mr);
    return static_cast<cudf::numeric_scalar<int64_t>*>(s.get())->value(stream);
  };
  std::vector<std::unique_ptr<cudf::table>> rows;
  for (int g = 0; g < groups; g++) {
    if (_group && count(partials.matches->view(), g) == 0) continue;
    std::vector<std::unique_ptr<cudf::column>> columns;
    if (_group)
      columns.push_back(
        std::make_unique<cudf::column>(cudf::slice(keys, {g, g + 1}, stream)[0], stream, mr));
    for (size_t k = 0; k < _sums.size(); k++) {
      auto value =
        cudf::reduce(part(partials.sums[k]->view(), g), *sum, values[k].type(), stream, mr);
      value->set_valid_async(count(partials.valid_counts[k]->view(), g) != 0, stream);
      columns.push_back(cudf::make_column_from_scalar(*value, 1, stream, mr));
    }
    rows.push_back(std::make_unique<cudf::table>(std::move(columns)));
  }
  std::unique_ptr<cudf::table> output;
  if (rows.empty())
    output = sirius::make_empty_table(output_types);
  else {
    std::vector<cudf::table_view> views;
    for (auto const& r : rows)
      views.push_back(r->view());
    output = cudf::concatenate(views, stream, mr);
  }
  SIRIUS_LOG_INFO(
    "[FUSED_JOIN_AGGREGATE] build_rows={} probe_rows={} groups={} sums={} original_build_child=1 "
    "group_side={}",
    bk->size(),
    pk->size(),
    groups,
    _sums.size(),
    _group_is_build ? "build" : "probe");
  return std::make_unique<pipelineable_operator_data>(
    std::vector<std::shared_ptr<::cucascade::data_batch>>{
      make_data_batch(std::move(output), *space, stream, batch_telemetry())});
}
}  // namespace sirius::op
