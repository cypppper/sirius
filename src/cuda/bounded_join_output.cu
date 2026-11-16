// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "op/bounded_join_output.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/copying.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/scan.h>

#include <algorithm>
#include <limits>

namespace sirius::op {
namespace {
__global__ void range_sizes(cudf::column_device_view probe,
                            int const* lower,
                            int const* upper,
                            bool nulls_equal,
                            int64_t* prefix)
{
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) prefix[0] = 0;
  if (i < probe.size()) {
    prefix[i + 1] = (nulls_equal || probe.is_valid(i)) ? int64_t(upper[i]) - lower[i] : 0;
  }
}
__global__ void expand_tile(int64_t const* prefix,
                            int const* lower,
                            int const* order,
                            int probe_rows,
                            int64_t offset,
                            int rows,
                            int* left,
                            int* right)
{
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows) return;
  int64_t rank = offset + i;
  int lo = 0, hi = probe_rows;
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (prefix[mid + 1] <= rank)
      lo = mid + 1;
    else
      hi = mid;
  }
  left[i]  = lo;
  right[i] = order[lower[lo] + rank - prefix[lo]];
}
}  // namespace

std::shared_ptr<bounded_join_index> make_bounded_join_index(cudf::table_view build_keys,
                                                            ::cuda::stream_ref stream,
                                                            rmm::device_async_resource_ref mr)
{
  auto result = std::make_shared<bounded_join_index>();
  result->order =
    cudf::sorted_order(build_keys, {cudf::order::ASCENDING}, {cudf::null_order::AFTER}, stream, mr);
  result->keys = cudf::gather(
    build_keys, result->order->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  return result;
}

bounded_join_stats consume_bounded_join(bounded_join_index const& index,
                                        cudf::table_view probe_keys,
                                        bool nulls_equal,
                                        int64_t tile_rows,
                                        bool full_output_control,
                                        bounded_pair_consumer const& consume,
                                        ::cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  auto lower = cudf::lower_bound(index.keys->view(),
                                 probe_keys,
                                 {cudf::order::ASCENDING},
                                 {cudf::null_order::AFTER},
                                 stream,
                                 mr);
  auto upper = cudf::upper_bound(index.keys->view(),
                                 probe_keys,
                                 {cudf::order::ASCENDING},
                                 {cudf::null_order::AFTER},
                                 stream,
                                 mr);
  auto probe = cudf::column_device_view::create(probe_keys.column(0), stream);
  auto n     = probe_keys.num_rows();
  rmm::device_uvector<int64_t> prefix(int64_t(n) + 1, stream, mr);
  range_sizes<<<std::max<int64_t>(1, (int64_t(n) + 255) / 256), 256, 0, stream.get()>>>(
    *probe, lower->view().data<int>(), upper->view().data<int>(), nulls_equal, prefix.data());
  CUDF_CUDA_TRY(cudaGetLastError());
  thrust::inclusive_scan(
    rmm::exec_policy(stream, mr), prefix.begin(), prefix.end(), prefix.begin());
  bounded_join_stats stats;
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    &stats.pairs, prefix.data() + n, sizeof(int64_t), cudaMemcpyDeviceToHost, stream.get()));
  stream.sync();
  if (full_output_control) {
    if (stats.pairs > std::numeric_limits<cudf::size_type>::max())
      throw std::runtime_error("range-full control exceeds cuDF row limit");
    tile_rows = std::max<int64_t>(1, stats.pairs);
  }
  if (tile_rows <= 0 || tile_rows > std::numeric_limits<cudf::size_type>::max())
    throw std::runtime_error("invalid bounded join tile size");
  int64_t offset = 0;
  do {
    auto rows = static_cast<int>(std::min(tile_rows, stats.pairs - offset));
    rmm::device_uvector<int> left(rows, stream, mr), right(rows, stream, mr);
    if (rows) {
      expand_tile<<<(int64_t(rows) + 255) / 256, 256, 0, stream.get()>>>(
        prefix.data(),
        lower->view().data<int>(),
        index.order->view().data<int>(),
        n,
        offset,
        rows,
        left.data(),
        right.data());
      CUDF_CUDA_TRY(cudaGetLastError());
    }
    consume(left, right);
    ++stats.tiles;
    stats.max_tile_rows = std::max<int64_t>(stats.max_tile_rows, rows);
    offset += rows;
  } while (offset < stats.pairs);
  return stats;
}
}  // namespace sirius::op
