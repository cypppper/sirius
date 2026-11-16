// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "expression/ast/node.hpp"
#include "op/sirius_physical_dense_count_join.hpp"
namespace sirius::op {
// Experimental SUM-join: reuses two-input FULL-barrier scheduling, not the
// dense-count execution algorithm. There are no multiplicity histograms.
class sirius_physical_fused_join_aggregate : public sirius_physical_dense_count_join {
 public:
  static constexpr auto TYPE = SiriusPhysicalOperatorType::FUSED_JOIN_AGGREGATE;
  struct sum_spec {
    std::unique_ptr<ast::node> value;
    std::unique_ptr<ast::node> predicate;
    bool conditional        = false;
    bool value_is_build     = false;
    bool predicate_is_build = false;
  };
  sirius_physical_fused_join_aggregate(duckdb::vector<logical_type> output_types,
                                       size_t build_key,
                                       size_t probe_key,
                                       std::vector<sum_spec> specs,
                                       std::unique_ptr<ast::node> group,
                                       bool group_is_build,
                                       size_t cardinality);
  partition_strategy get_partition_strategy(partition_sizing_input const&) override;
  std::unique_ptr<operator_data> execute(operator_data const&, ::cuda::stream_ref) override;
  std::size_t no_history_peak_memory_estimate(input_stats const&) const override;
  std::string params_to_string() const override;

 private:
  std::vector<sum_spec> _sums;
  std::unique_ptr<ast::node> _group;
  bool _group_is_build;
};
// Runs before physical-schema compression and pipeline splitting. Opt-in only:
// SIRIUS_EXP_JOIN_AGGREGATION=1. Ordinary SQL is unchanged when unset.
void fuse_join_aggregates(duckdb::unique_ptr<sirius_physical_operator>& root);
}  // namespace sirius::op
