// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "expression/ast/node.hpp"
#include "op/sirius_physical_operator.hpp"

#include <cudf/table/table.hpp>

#include <rmm/device_uvector.hpp>

#include <functional>
namespace sirius::op {
using join_map_consumer =
  std::function<std::unique_ptr<cudf::table>(cudf::table_view,
                                             cudf::table_view,
                                             rmm::device_uvector<cudf::size_type> const&,
                                             rmm::device_uvector<cudf::size_type> const&,
                                             ::cuda::stream_ref,
                                             rmm::device_async_resource_ref)>;
struct join_map_sum_argument {
  std::unique_ptr<ast::node> expression;
  bool from_right;
  cudf::data_type output_type;
};
join_map_consumer make_join_map_sum_consumer(std::vector<join_map_sum_argument> arguments);
void fuse_join_map_sums(duckdb::unique_ptr<sirius_physical_operator>& root);
}  // namespace sirius::op
