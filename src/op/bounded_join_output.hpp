// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

#include <rmm/device_uvector.hpp>

#include <functional>

namespace sirius::op {
// Experimental sorted build index. No key multiplicities replace join matches:
// every matching pair is enumerated and consumed by ordinary vector operators.
struct bounded_join_index {
  std::unique_ptr<cudf::column> order;
  std::unique_ptr<cudf::table> keys;
};
std::shared_ptr<bounded_join_index> make_bounded_join_index(cudf::table_view build_keys,
                                                            ::cuda::stream_ref stream,
                                                            rmm::device_async_resource_ref mr);
struct bounded_join_stats {
  int64_t pairs         = 0;
  int64_t tiles         = 0;
  int64_t max_tile_rows = 0;
};
using bounded_pair_consumer = std::function<void(rmm::device_uvector<cudf::size_type> const&,
                                                 rmm::device_uvector<cudf::size_type> const&)>;
bounded_join_stats consume_bounded_join(bounded_join_index const& index,
                                        cudf::table_view probe_keys,
                                        bool nulls_equal,
                                        int64_t tile_rows,
                                        bool full_output_control,
                                        bounded_pair_consumer const& consume,
                                        ::cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr);
}  // namespace sirius::op
