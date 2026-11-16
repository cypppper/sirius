// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "expression/ast/utils.hpp"
#include "expression_evaluator/gpu_expression_translator_internal.hpp"
#include "log/logging.hpp"
#include "op/aggregate/join_map_sum.hpp"
#include "op/aggregate/lazy_join_expression.hpp"
#include "op/sirius_physical_fused_join_aggregate.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_projection.hpp"
#include "op/sirius_physical_ungrouped_aggregate.hpp"

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <type_traits>
namespace sirius::op {
namespace {
bool zero(ast::node const& expr)
{
  if (!expr.holds<ast::constant>()) return false;
  return std::visit(
    [](auto const& v) {
      using T = std::decay_t<decltype(v)>;
      if constexpr (std::is_arithmetic_v<T>)
        return v == 0;
      else if constexpr (std::is_same_v<T, decimal32> || std::is_same_v<T, decimal64> ||
                         std::is_same_v<T, decimal128>)
        return v.value == 0;
      else
        return false;
    },
    expr.get<ast::constant>().payload);
}
bool narrow_value(logical_type const& type)
{
  return (type.is_decimal() && type.decimal_precision() <= 18) || type.id() == type_id::TINYINT ||
         type.id() == type_id::SMALLINT || type.id() == type_id::INTEGER ||
         type.id() == type_id::BIGINT;
}
void attempt(duckdb::unique_ptr<sirius_physical_operator>& slot)
{
  bool grouped = slot->type == SiriusPhysicalOperatorType::HASH_GROUP_BY;
  if (slot->type != SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE && !grouped) return;
  if (slot->children.size() != 1) return;
  std::vector<std::unique_ptr<ast::node>> values;
  std::unique_ptr<ast::node> group;
  if (grouped) {
    auto const& ag = slot->Cast<sirius_physical_grouped_aggregate>();
    if (ag.has_avg || ag.has_count_distinct || ag.has_grouping_sets() || ag.group_idx.size() != 1 ||
        ag.cudf_aggregates.size() + 1 != slot->types.size() ||
        ag.cudf_aggregate_idx.size() != ag.cudf_aggregates.size())
      return;
    for (size_t i = 0; i < ag.cudf_aggregates.size(); i++) {
      if (ag.cudf_aggregates[i] != cudf::aggregation::SUM || ag.cudf_aggregate_idx[i] < 0) return;
      int col = ag.cudf_aggregate_idx[i];
      values.push_back(std::make_unique<ast::node>(
        ast::reference{static_cast<uint32_t>(col), slot->children[0]->types.at(col)}));
    }
    int col = ag.group_idx[0];
    if (col < 0) return;
    group = std::make_unique<ast::node>(
      ast::reference{static_cast<uint32_t>(col), slot->children[0]->types.at(col)});
  } else {
    for (auto const& e : slot->Cast<sirius_physical_ungrouped_aggregate>().aggregates) {
      if (!e->holds<ast::aggregate>()) return;
      auto const& a = e->get<ast::aggregate>();
      if (a.distinct() || a.arguments().size() != 1 ||
          (a.function() != aggregate_id::sum && a.function() != aggregate_id::sum_no_overflow))
        return;
      values.push_back(ast::clone(*a.arguments()[0]));
    }
  }
  if (values.empty() || values.size() > 4) return;
  for (size_t i = 0; i < values.size(); i++) {
    auto const& t = slot->types[i + (grouped ? 1 : 0)];
    if (!t.is_decimal() && t.id() != type_id::HUGEINT) return;
  }
  auto* input = slot->children[0].get();
  while (input->type == SiriusPhysicalOperatorType::PROJECTION) {
    auto const& p = input->Cast<sirius_physical_projection>();
    for (auto& v : values)
      v = ast::substitute_references(*v, p.select_list);
    if (group) group = ast::substitute_references(*group, p.select_list);
    if (input->children.size() != 1) return;
    input = input->children[0].get();
  }
  if (input->type != SiriusPhysicalOperatorType::HASH_JOIN) return;
  auto& join = input->Cast<sirius_physical_hash_join>();
  if (join.join_type != duckdb::JoinType::INNER || join.conditions.size() != 1 ||
      join.conditions[0].comparison != comparison_type::equal || join.children.size() != 2 ||
      !join.conditions[0].left->holds<ast::reference>() ||
      !join.conditions[0].right->holds<ast::reference>())
    return;
  std::vector<int> side;
  std::vector<std::unique_ptr<ast::node>> remap;
  for (int s = 0; s < 2; s++) {
    auto const& cols = s == 0 ? join.lhs_output_columns.col_idxs : join.rhs_output_columns.col_idxs;
    for (auto c : cols) {
      side.push_back(s);
      remap.push_back(std::make_unique<ast::node>(
        ast::reference{static_cast<uint32_t>(c), join.children[s]->types.at(c)}));
    }
  }
  auto pure = [&](ast::node const& n, int s) {
    bool ok = true;
    ast::visit_references(n, [&](ast::reference const& r) {
      if (r.column_index >= side.size() || side[r.column_index] != s) ok = false;
    });
    return ok;
  };
  // Ordinary Sirius hashes children[1] and probes children[0]. Preserve that
  // physical choice; expression placement must not reorder the join inputs.
  constexpr int build = 1, probe = 0;
  auto source_side = [&](ast::node const& n) -> std::optional<int> {
    // A constant can be evaluated on either input. Prefer the build side.
    if (pure(n, build)) return build;
    if (pure(n, probe)) return probe;
    return std::nullopt;
  };
  auto group_side = group ? source_side(*group) : std::optional<int>{probe};
  if (!group_side) return;
  std::vector<sirius_physical_fused_join_aggregate::sum_spec> specs;
  for (auto const& n : values) {
    sirius_physical_fused_join_aggregate::sum_spec s;
    if (n->holds<ast::case_expr>()) {
      auto const& c = n->get<ast::case_expr>();
      if (c.eager || c.cases.size() != 1 || !c.else_ || !zero(*c.else_) ||
          !narrow_value(c.cases[0].then_->return_type()))
        return;
      auto value_side     = source_side(*c.cases[0].then_);
      auto predicate_side = source_side(*c.cases[0].when_);
      if (!value_side || !predicate_side) return;
      s.value              = ast::substitute_references(*c.cases[0].then_, remap);
      s.predicate          = ast::substitute_references(*c.cases[0].when_, remap);
      s.conditional        = true;
      s.value_is_build     = *value_side == build;
      s.predicate_is_build = *predicate_side == build;
    } else {
      auto side = source_side(*n);
      if (!side || !narrow_value(n->return_type())) return;
      s.value          = ast::substitute_references(*n, remap);
      s.value_is_build = *side == build;
    }
    if (!compile_lazy_join_expression(*s.value) ||
        (s.predicate && !compile_lazy_join_expression(*s.predicate)))
      return;
    specs.push_back(std::move(s));
  }
  if (grouped && (specs.size() != 2 || std::any_of(specs.begin(), specs.end(), [](auto const& s) {
                    return !s.conditional;
                  })))
    return;
  auto bi  = join.conditions[0].right->get<ast::reference>().column_index;
  auto pi  = join.conditions[0].left->get<ast::reference>().column_index;
  auto kt0 = join.children[build]->types.at(bi).id(), kt1 = join.children[probe]->types.at(pi).id();
  auto key_ok = [](type_id t) { return t == type_id::INTEGER || t == type_id::BIGINT; };
  if (!key_ok(kt0) || !key_ok(kt1)) return;
  auto fused = duckdb::make_uniq<sirius_physical_fused_join_aggregate>(
    slot->types,
    bi,
    pi,
    std::move(specs),
    group ? ast::substitute_references(*group, remap) : nullptr,
    *group_side == build,
    slot->estimated_cardinality);
  // The fused scheduling base stores build first; input identities remain those
  // chosen by the original HASH_JOIN, even though its child numbering differs.
  fused->children.push_back(std::move(join.children[build]));
  fused->children.push_back(std::move(join.children[probe]));
  SIRIUS_LOG_INFO(
    "[planner] Fusing inner join and {} SUMs, grouped={}, preserving original build child=1",
    values.size(),
    grouped);
  slot = std::move(fused);
}
void walk(duckdb::unique_ptr<sirius_physical_operator>& slot)
{
  if (!slot) return;
  attempt(slot);
  for (auto& c : slot->children)
    walk(c);
}
}  // namespace
void fuse_join_aggregates(duckdb::unique_ptr<sirius_physical_operator>& root)
{
  auto* flag = std::getenv("SIRIUS_EXP_JOIN_AGGREGATION");
  if (flag && std::string_view(flag) == "1") walk(root);
}
namespace {
void attempt_map_sum(duckdb::unique_ptr<sirius_physical_operator>& slot)
{
  if (slot->type != SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE || slot->children.size() != 1)
    return;
  auto& aggregate = slot->Cast<sirius_physical_ungrouped_aggregate>();
  std::vector<std::unique_ptr<ast::node>> expressions;
  for (auto const& root : aggregate.aggregates) {
    if (!root->holds<ast::aggregate>()) return;
    auto const& sum = root->get<ast::aggregate>();
    if (sum.distinct() || sum.arguments().size() != 1 ||
        (sum.function() != aggregate_id::sum && sum.function() != aggregate_id::sum_no_overflow))
      return;
    expressions.push_back(ast::clone(*sum.arguments()[0]));
  }
  if (expressions.empty()) return;
  auto* child = slot->children[0].get();
  while (child->type == SiriusPhysicalOperatorType::PROJECTION) {
    for (auto& expression : expressions)
      expression = ast::substitute_references(
        *expression, child->Cast<sirius_physical_projection>().select_list);
    if (child->children.size() != 1) return;
    child = child->children[0].get();
  }
  if (child->type != SiriusPhysicalOperatorType::HASH_JOIN) return;
  auto& join = child->Cast<sirius_physical_hash_join>();
  if (join.join_type != duckdb::JoinType::INNER || join.matched_pair_consumer) return;
  // This experiment runs after schema compression. Do not invalidate an existing
  // physical output contract when replacing joined columns with wide SUM states.
  if (join.has_physical_overrides() || slot->has_physical_overrides()) return;
  std::vector<int> sides;
  std::vector<std::unique_ptr<ast::node>> references;
  for (int side = 0; side < 2; ++side) {
    auto const& columns =
      side ? join.rhs_output_columns.col_idxs : join.lhs_output_columns.col_idxs;
    for (auto index : columns) {
      sides.push_back(side);
      references.push_back(std::make_unique<ast::node>(
        ast::reference{static_cast<uint32_t>(index), join.children[side]->types.at(index)}));
    }
  }
  std::vector<join_map_sum_argument> arguments;
  gpu_expression_translator translator(cudf::get_default_stream(),
                                       cudf::get_current_device_resource_ref());
  for (size_t k = 0; k < expressions.size(); ++k) {
    if (!narrow_value(expressions[k]->return_type())) return;
    auto output = require_cudf_type(slot->types[k]);
    if (output.id() != cudf::type_id::DECIMAL128) return;
    int selected = -1;
    bool mixed   = false;
    ast::visit_references(*expressions[k], [&](ast::reference const& ref) {
      if (ref.column_index >= sides.size()) {
        mixed = true;
        return;
      }
      int side = sides[ref.column_index];
      if (selected != -1 && selected != side) mixed = true;
      selected = side;
    });
    if (mixed) return;
    if (selected < 0) selected = 0;
    auto remapped = ast::substitute_references(*expressions[k], references);
    if (!translator.translate_expression(
          *remapped,
          selected ? cudf::ast::table_reference::RIGHT : cudf::ast::table_reference::LEFT))
      return;
    arguments.push_back({std::move(remapped), selected == 1, output});
  }
  // Keep the existing local SUM and merge operators. They reduce the compact
  // per-warp states instead of M materialized join rows, including NULL states.
  for (size_t k = 0; k < aggregate.aggregates.size(); ++k) {
    std::vector<std::unique_ptr<ast::node>> refs;
    refs.push_back(
      std::make_unique<ast::node>(ast::reference{static_cast<uint32_t>(k), slot->types[k]}));
    aggregate.aggregates[k] = std::make_unique<ast::node>(
      ast::aggregate{aggregate_id::sum, std::move(refs), slot->types[k], false});
  }
  aggregate.aggregate_input_max_abs.assign(expressions.size(), std::nullopt);
  join.matched_pair_consumer = make_join_map_sum_consumer(std::move(arguments));
  join.types                 = slot->types;
  std::function<duckdb::unique_ptr<sirius_physical_operator>(
    duckdb::unique_ptr<sirius_physical_operator>)>
    remove_projection;
  remove_projection = [&](duckdb::unique_ptr<sirius_physical_operator> node)
    -> duckdb::unique_ptr<sirius_physical_operator> {
    if (node->type == SiriusPhysicalOperatorType::PROJECTION)
      return remove_projection(std::move(node->children[0]));
    return node;
  };
  slot->children[0] = remove_projection(std::move(slot->children[0]));
  SIRIUS_LOG_INFO(
    "[planner] Join-map SUM consumer enabled; ordinary join scheduling and cuDF expression "
    "evaluator retained");
}
void walk_map_sums(duckdb::unique_ptr<sirius_physical_operator>& slot)
{
  if (!slot) return;
  attempt_map_sum(slot);
  for (auto& child : slot->children)
    walk_map_sums(child);
}
}  // namespace
void fuse_join_map_sums(duckdb::unique_ptr<sirius_physical_operator>& root)
{
  auto* flag = std::getenv("SIRIUS_EXP_JOIN_MAP_SUM");
  if (flag && std::string_view(flag) == "1") walk_map_sums(root);
}
}  // namespace sirius::op
