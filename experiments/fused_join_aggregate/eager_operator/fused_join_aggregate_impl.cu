// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "op/aggregate/fused_join_aggregate_impl.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include <array>
#include <limits>
#include <stdexcept>
namespace sirius::op {
namespace {
constexpr int max_aggregates = 4;
struct device_column {
  void const* head               = nullptr;
  cudf::bitmask_type const* mask = nullptr;
  int n = 0, offset = 0;
  __device__ int size() const { return n; }
  __device__ bool is_valid(int row) const
  {
    int bit = row + offset;
    return !mask || ((mask[bit / 32] >> (bit % 32)) & 1);
  }
  template <class T>
  __device__ T const* data() const
  {
    return static_cast<T const*>(head) + offset;
  }
};
device_column view(cudf::column_view c) { return {c.head(), c.null_mask(), c.size(), c.offset()}; }
struct arguments {
  device_column bk, pk, group;
  device_column values[max_aggregates], predicates[max_aggregates];
  __int128_t* sums[max_aggregates];
  int64_t* valid[max_aggregates];
  bool conditional[max_aggregates], has_predicate[max_aggregates], has_group;
  bool value_is_build[max_aggregates], predicate_is_build[max_aggregates], group_is_build;
  int64_t* matches;
  int aggregates, groups, warps;
};
__device__ unsigned hash_key(int64_t key, unsigned mask)
{
  uint64_t x = static_cast<uint64_t>(key);
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return static_cast<unsigned>(x) & mask;
}
__global__ void build_keys(device_column keys, int* heads, int* next, unsigned mask)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < keys.size() && keys.is_valid(i))
    next[i] = atomicExch(heads + hash_key(keys.data<int64_t>()[i], mask), i);
}
__device__ __int128_t warp_sum(__int128_t value)
{
  for (int d = 16; d; d /= 2) {
    auto lo = static_cast<uint64_t>(value);
    auto hi = static_cast<uint64_t>(value >> 64);
    lo      = __shfl_down_sync(0xffffffff, lo, d);
    hi      = __shfl_down_sync(0xffffffff, hi, d);
    value += static_cast<__int128_t>((static_cast<__uint128_t>(hi) << 64) | lo);
  }
  return value;
}
__global__ void probe(arguments a, int const* heads, int const* next, unsigned mask)
{
  int i       = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = i < a.pk.size() && a.pk.is_valid(i);
  int64_t key = active ? a.pk.data<int64_t>()[i] : 0;
  int first   = active ? heads[hash_key(key, mask)] : -1;
  int warp    = i / 32;
  for (int group = 0; group < a.groups; group++) {
    // A probe can match build rows belonging to different groups. Choose a
    // group per matching pair, never once per probe. Traversing the short chain
    // per bounded group avoids a groups x aggregates array of thread-local sums.
    __int128_t sums[max_aggregates] = {};
    int64_t valid[max_aggregates]   = {};
    int64_t count                   = 0;
    for (int j = first; j != -1; j = next[j]) {
      if (a.bk.data<int64_t>()[j] != key) continue;
      int g = a.has_group ? a.group.data<int32_t>()[a.group_is_build ? j : i] : 0;
      if (g != group) continue;
      ++count;
      for (int k = 0; k < a.aggregates; k++) {
        int predicate_row = a.predicate_is_build[k] ? j : i;
        bool keep         = !a.has_predicate[k] || (a.predicates[k].is_valid(predicate_row) &&
                                            a.predicates[k].data<bool>()[predicate_row]);
        if (!keep) {
          if (a.conditional[k]) ++valid[k];
          continue;
        }
        int value_row = a.value_is_build[k] ? j : i;
        if (a.values[k].is_valid(value_row)) {
          sums[k] += a.values[k].data<__int128_t>()[value_row];
          ++valid[k];
        }
      }
    }
    for (int d = 16; d; d /= 2)
      count += __shfl_down_sync(0xffffffff, count, d);
    if (threadIdx.x % 32 == 0) a.matches[group * a.warps + warp] = count;
    for (int k = 0; k < a.aggregates; k++) {
      auto sum  = warp_sum(sums[k]);
      int64_t n = valid[k];
      for (int d = 16; d; d /= 2)
        n += __shfl_down_sync(0xffffffff, n, d);
      if (threadIdx.x % 32 == 0) {
        a.sums[k][group * a.warps + warp]  = sum;
        a.valid[k][group * a.warps + warp] = n;
      }
    }
  }
}
}  // namespace
fused_join_partials probe_sum_join(cudf::column_view bk,
                                   cudf::column_view pk,
                                   std::vector<cudf::column_view> const& values,
                                   std::vector<cudf::column_view> const& predicates,
                                   std::vector<bool> const& conditional,
                                   std::vector<bool> const& value_is_build,
                                   std::vector<bool> const& predicate_is_build,
                                   cudf::column_view const* group_ids,
                                   bool group_is_build,
                                   int groups,
                                   ::cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  if (values.empty() || values.size() > max_aggregates || groups < 1 || groups > 4)
    throw std::runtime_error("experimental fused SUM join: unsupported state size");
  unsigned buckets = 1;
  while (buckets < static_cast<uint64_t>(bk.size()) * 2) {
    if (buckets > (1U << 29)) throw std::runtime_error("experimental fused join build too large");
    buckets *= 2;
  }
  rmm::device_uvector<int> heads(buckets, stream, mr), next(bk.size(), stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(heads.data(), 255, buckets * sizeof(int), stream.get()));
  if (bk.size())
    build_keys<<<(bk.size() + 255) / 256, 256, 0, stream.get()>>>(
      view(bk), heads.data(), next.data(), buckets - 1);
  int blocks = std::max(1, (pk.size() + 255) / 256);
  int warps  = blocks * 8;
  fused_join_partials out;
  out.groups = groups;
  out.warps  = warps;
  arguments a{};
  a.bk             = view(bk);
  a.pk             = view(pk);
  a.aggregates     = values.size();
  a.groups         = groups;
  a.warps          = warps;
  a.has_group      = group_ids != nullptr;
  a.group_is_build = group_is_build;
  if (group_ids) a.group = view(*group_ids);
  for (size_t k = 0; k < values.size(); k++) {
    out.sums.push_back(cudf::make_fixed_width_column(
      values[k].type(), groups * warps, cudf::mask_state::UNALLOCATED, stream, mr));
    out.valid_counts.push_back(cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                                         groups * warps,
                                                         cudf::mask_state::UNALLOCATED,
                                                         stream,
                                                         mr));
    a.values[k]        = view(values[k]);
    a.has_predicate[k] = predicates[k].type().id() != cudf::type_id::EMPTY;
    if (a.has_predicate[k]) a.predicates[k] = view(predicates[k]);
    a.sums[k]               = out.sums[k]->mutable_view().data<__int128_t>();
    a.valid[k]              = out.valid_counts[k]->mutable_view().data<int64_t>();
    a.conditional[k]        = conditional[k];
    a.value_is_build[k]     = value_is_build[k];
    a.predicate_is_build[k] = predicate_is_build[k];
  }
  out.matches = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                          groups * warps,
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);
  a.matches   = out.matches->mutable_view().data<int64_t>();
  probe<<<blocks, 256, 0, stream.get()>>>(a, heads.data(), next.data(), buckets - 1);
  CUDF_CUDA_TRY(cudaGetLastError());
  return out;
}
}  // namespace sirius::op
