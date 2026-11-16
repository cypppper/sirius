// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "op/sirius_physical_fused_join_aggregate.hpp"

#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "expression_evaluator/expression_evaluator.hpp"
#include "log/logging.hpp"
#include "op/aggregate/fused_join_aggregate_impl.hpp"
#include "op/aggregate/lazy_join_expression.hpp"
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
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
namespace sirius::op {
namespace {
int decimal_scale(logical_type const& type) { return type.is_decimal() ? type.decimal_scale() : 0; }
bool numeric_bounds(logical_type const& type, lazy_join_instruction& instruction)
{
  if (type.is_decimal()) {
    if (type.decimal_precision() > 18) return false;
    __int128_t limit = 1;
    for (int i = 0; i < type.decimal_precision(); ++i)
      limit *= 10;
    instruction.maximum = limit - 1;
    instruction.minimum = -instruction.maximum;
  } else {
    int bits = 0;
    switch (type.id()) {
      case type_id::BOOLEAN:
        instruction.minimum = 0;
        instruction.maximum = 1;
        return true;
      case type_id::TINYINT: bits = 8; break;
      case type_id::SMALLINT: bits = 16; break;
      case type_id::INTEGER: bits = 32; break;
      case type_id::BIGINT: bits = 64; break;
      default: return false;
    }
    instruction.maximum = static_cast<__int128_t>((__uint128_t{1} << (bits - 1)) - 1);
    instruction.minimum = -instruction.maximum - 1;
  }
  return true;
}
bool emit_lazy(ast::node const& node, lazy_join_expression& program, int depth)
{
  if (depth > lazy_join_max_instructions ||
      program.instructions.size() >= lazy_join_max_instructions)
    return false;
  lazy_join_instruction instruction;
  instruction.scale = decimal_scale(node.return_type());
  auto emit         = [&] {
    program.instructions.push_back(instruction);
    return true;
  };
  auto child = [&](ast::node const& n) { return emit_lazy(n, program, depth + 1); };
  auto string_leaf =
    [&](ast::node const& lhs, ast::node const& rhs, lazy_join_opcode opcode, bool like) {
      if (!lhs.holds<ast::reference>() || !rhs.holds<ast::constant>() ||
          lhs.return_type().id() != type_id::VARCHAR)
        return false;
      auto const& payload = rhs.get<ast::constant>().payload;
      instruction.op      = opcode;
      instruction.column  = lhs.get<ast::reference>().column_index;
      if (std::holds_alternative<null_value>(payload)) {
        instruction.valid = false;
        return emit();
      }
      if (!std::holds_alternative<std::string>(payload)) return false;
      auto text = std::get<std::string>(payload);
      if (like) {
        if (text.empty() || text.back() != '%') return false;
        text.pop_back();
        if (text.find_first_of("%_\\") != std::string::npos ||
            std::any_of(text.begin(), text.end(), [](unsigned char c) { return c >= 128; }))
          return false;
      }
      if (text.size() > lazy_join_literal_bytes) return false;
      instruction.text_size = text.size();
      std::memcpy(instruction.text, text.data(), text.size());
      return emit();
    };
  if (node.holds<ast::reference>()) {
    if (!numeric_bounds(node.return_type(), instruction)) return false;
    instruction.op     = lazy_join_opcode::column;
    instruction.column = node.get<ast::reference>().column_index;
    return emit();
  }
  if (node.holds<ast::constant>()) {
    if (!numeric_bounds(node.return_type(), instruction)) return false;
    instruction.op = lazy_join_opcode::literal;
    bool supported = true;
    std::visit(
      [&](auto const& value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, null_value>)
          instruction.valid = false;
        else if constexpr (std::is_integral_v<T>)
          instruction.number = value;
        else if constexpr (std::is_same_v<T, decimal32> || std::is_same_v<T, decimal64> ||
                           std::is_same_v<T, decimal128>)
          instruction.number = value.value;
        else
          supported = false;
      },
      node.get<ast::constant>().payload);
    return supported && emit();
  }
  if (node.holds<ast::cast>()) {
    auto const& cast = node.get<ast::cast>();
    // Narrowing/rescaling casts require the engine's complete conversion policy.
    bool restore = cast.kind == ast::cast_kind::carrier_restore;
    if (cast.try_cast || !numeric_bounds(cast.target_type, instruction) ||
        (!restore && instruction.scale < decimal_scale(cast.child->return_type())) ||
        !child(*cast.child))
      return false;
    instruction.op        = lazy_join_opcode::cast;
    instruction.lhs_scale = restore ? instruction.scale : decimal_scale(cast.child->return_type());
    return emit();
  }
  if (node.holds<ast::comparison>()) {
    auto const& comparison = node.get<ast::comparison>();
    if (comparison.op != comparison_type::equal && comparison.op != comparison_type::not_equal)
      return false;
    bool equal = comparison.op == comparison_type::equal;
    if (comparison.left->return_type().id() == type_id::VARCHAR ||
        comparison.right->return_type().id() == type_id::VARCHAR) {
      auto op = equal ? lazy_join_opcode::string_equal : lazy_join_opcode::string_not_equal;
      return string_leaf(*comparison.left, *comparison.right, op, false) ||
             string_leaf(*comparison.right, *comparison.left, op, false);
    }
    if (!child(*comparison.left) || !child(*comparison.right)) return false;
    instruction.op        = equal ? lazy_join_opcode::equal : lazy_join_opcode::not_equal;
    instruction.lhs_scale = decimal_scale(comparison.left->return_type());
    instruction.rhs_scale = decimal_scale(comparison.right->return_type());
    instruction.scale     = std::max(instruction.lhs_scale, instruction.rhs_scale);
    return emit();
  }
  if (node.holds<ast::conjunction>()) {
    auto const& conjunction = node.get<ast::conjunction>();
    if (conjunction.children.empty() || conjunction.op == ast::conjunction::kind::invalid)
      return false;
    instruction.op = conjunction.op == ast::conjunction::kind::op_and
                       ? lazy_join_opcode::logical_and
                       : lazy_join_opcode::logical_or;
    if (!child(*conjunction.children[0])) return false;
    for (size_t i = 1; i < conjunction.children.size(); ++i)
      if (!child(*conjunction.children[i]) || !emit()) return false;
    return true;
  }
  if (node.holds<ast::unary_op>()) {
    auto const& unary = node.get<ast::unary_op>();
    if (unary.op != ast::unary_op::kind::op_not || !child(*unary.child)) return false;
    instruction.op = lazy_join_opcode::logical_not;
    return emit();
  }
  if (node.holds<ast::in_list>()) {
    auto const& in = node.get<ast::in_list>();
    if (in.values.empty()) return false;
    for (size_t i = 0; i < in.values.size(); ++i) {
      if (in.probe->return_type().id() == type_id::VARCHAR) {
        if (!string_leaf(*in.probe, *in.values[i], lazy_join_opcode::string_equal, false))
          return false;
      } else {
        if (!child(*in.probe) || !child(*in.values[i])) return false;
        instruction.op        = lazy_join_opcode::equal;
        instruction.lhs_scale = decimal_scale(in.probe->return_type());
        instruction.rhs_scale = decimal_scale(in.values[i]->return_type());
        instruction.scale     = std::max(instruction.lhs_scale, instruction.rhs_scale);
        emit();
      }
      if (i) {
        instruction.op = lazy_join_opcode::logical_or;
        emit();
      }
    }
    if (in.negated) {
      instruction.op = lazy_join_opcode::logical_not;
      emit();
    }
    return true;
  }
  if (node.holds<ast::function_call>()) {
    auto const& call = node.get<ast::function_call>();
    auto const& args = call.arguments();
    if (args.size() != 2) return false;
    if (call.function() == function_id::prefix || call.function() == function_id::like)
      return string_leaf(
        *args[0], *args[1], lazy_join_opcode::string_prefix, call.function() == function_id::like);
    if (!numeric_bounds(call.return_type(), instruction)) return false;
    switch (call.function()) {
      case function_id::add: instruction.op = lazy_join_opcode::add; break;
      case function_id::sub: instruction.op = lazy_join_opcode::sub; break;
      case function_id::mul: instruction.op = lazy_join_opcode::mul; break;
      default: return false;
    }
    instruction.lhs_scale = decimal_scale(args[0]->return_type());
    instruction.rhs_scale = decimal_scale(args[1]->return_type());
    if (instruction.scale < instruction.lhs_scale || instruction.scale < instruction.rhs_scale)
      return false;
    if (instruction.op == lazy_join_opcode::mul &&
        instruction.scale != instruction.lhs_scale + instruction.rhs_scale)
      return false;
    return child(*args[0]) && child(*args[1]) && emit();
  }
  return false;
}
}  // namespace
std::optional<lazy_join_expression> compile_lazy_join_expression(ast::node const& node)
{
  lazy_join_expression program;
  if (!emit_lazy(node, program, 0) || program.instructions.size() > lazy_join_max_instructions)
    return std::nullopt;
  int stack = 0;
  for (auto const& instruction : program.instructions) {
    switch (instruction.op) {
      case lazy_join_opcode::column:
      case lazy_join_opcode::literal:
      case lazy_join_opcode::string_equal:
      case lazy_join_opcode::string_not_equal:
      case lazy_join_opcode::string_prefix: ++stack; break;
      case lazy_join_opcode::cast:
      case lazy_join_opcode::logical_not: break;
      default: --stack; break;
    }
    if (stack < 1 || stack > lazy_join_stack_size) return std::nullopt;
  }
  if (stack != 1) return std::nullopt;
  return program;
}
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
  auto* lazy_flag = std::getenv("SIRIUS_EXP_JOIN_AGGREGATION_LAZY");
  bool lazy       = !lazy_flag || std::string_view(lazy_flag) != "0";
  lazy_join_inputs lazy_inputs;
  lazy_inputs.build = build->view();
  lazy_inputs.probe = probe->view();
  auto* audit_flag  = std::getenv("SIRIUS_EXP_JOIN_AGGREGATION_AUDIT");
  lazy_inputs.audit = audit_flag && std::string_view(audit_flag) == "1";
  for (size_t k = 0; k < _sums.size(); k++) {
    auto dt = require_cudf_type(types[(_group ? 1 : 0) + k]);
    conditional.push_back(_sums[k].conditional);
    value_is_build.push_back(_sums[k].value_is_build);
    predicate_is_build.push_back(_sums[k].predicate_is_build);
    predicates.emplace_back();
    if (lazy) {
      auto value = compile_lazy_join_expression(*_sums[k].value);
      if (!value) throw std::runtime_error("lazy join: unsupported value after schema compression");
      lazy_inputs.values.push_back(std::move(*value));
      lazy_inputs.predicates.emplace_back();
      if (_sums[k].predicate) {
        auto predicate = compile_lazy_join_expression(*_sums[k].predicate);
        if (!predicate)
          throw std::runtime_error("lazy join: unsupported predicate after schema compression");
        lazy_inputs.predicates.back() = std::move(*predicate);
      }
      // Only the output type is needed; no per-input aggregate column is created.
      values.emplace_back(dt, 0, nullptr, nullptr, 0);
    } else {
      expression_evaluator eval(*_sums[k].value, mr, stream, expression_policy());
      expression_results.push_back(
        eval.evaluate(_sums[k].value_is_build ? build->view() : probe->view()));
      cast_values.push_back(
        cudf::cast(expression_results.back()->view().column(0), dt, stream, mr));
      values.push_back(cast_values.back()->view());
      if (_sums[k].predicate) {
        expression_evaluator pred(*_sums[k].predicate, mr, stream, expression_policy());
        expression_results.push_back(
          pred.evaluate(_sums[k].predicate_is_build ? build->view() : probe->view()));
        predicates.back() = expression_results.back()->view().column(0);
      }
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
                                 mr,
                                 lazy ? &lazy_inputs : nullptr);
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
  if (partials.errors) {
    auto error = cudf::get_element(partials.errors->view(), 0, stream, mr);
    if (static_cast<cudf::numeric_scalar<int32_t>*>(error.get())->value(stream))
      throw std::runtime_error("lazy join aggregate expression overflow");
  }
  if (partials.expression_evaluations) {
    auto predicates = cudf::get_element(partials.expression_evaluations->view(), 0, stream, mr);
    auto values     = cudf::get_element(partials.expression_evaluations->view(), 1, stream, mr);
    SIRIUS_LOG_INFO("[LAZY_EXPR_AUDIT] predicate_evaluations={} value_evaluations={}",
                    static_cast<cudf::numeric_scalar<int64_t>*>(predicates.get())->value(stream),
                    static_cast<cudf::numeric_scalar<int64_t>*>(values.get())->value(stream));
  }
  SIRIUS_LOG_INFO(
    "[FUSED_JOIN_AGGREGATE] build_rows={} probe_rows={} groups={} sums={} original_build_child=1 "
    "group_side={} lazy={}",
    bk->size(),
    pk->size(),
    groups,
    _sums.size(),
    _group_is_build ? "build" : "probe",
    lazy);
  return std::make_unique<pipelineable_operator_data>(
    std::vector<std::shared_ptr<::cucascade::data_batch>>{
      make_data_batch(std::move(output), *space, stream, batch_telemetry())});
}
}  // namespace sirius::op
