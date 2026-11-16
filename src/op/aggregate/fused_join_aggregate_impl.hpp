// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "op/aggregate/lazy_join_expression.hpp"

#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <memory>
#include <vector>
namespace sirius::op {
struct fused_join_partials {
  std::vector<std::unique_ptr<cudf::column>> sums;
  std::vector<std::unique_ptr<cudf::column>> valid_counts;
  std::unique_ptr<cudf::column> matches;
  int groups;
  int warps;
  std::vector<lazy_join_column> host_columns;
  std::unique_ptr<cudf::column> expression_evaluations, errors;
};
// Values have DECIMAL128 carriers; predicates are BOOL8 (null means false).
// All sums happen after equality matching. Duplicates are traversed individually.
fused_join_partials probe_sum_join(cudf::column_view build_keys,
                                   cudf::column_view probe_keys,
                                   std::vector<cudf::column_view> const& values,
                                   std::vector<cudf::column_view> const& predicates,
                                   std::vector<bool> const& conditional,
                                   std::vector<bool> const& value_is_build,
                                   std::vector<bool> const& predicate_is_build,
                                   cudf::column_view const* group_ids,
                                   bool group_is_build,
                                   int groups,
                                   ::cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr,
                                   lazy_join_inputs const* lazy = nullptr);
}  // namespace sirius::op
