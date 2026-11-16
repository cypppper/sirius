// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "op/aggregate/fused_join_aggregate_impl.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
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
struct lazy_device_program {
  lazy_join_instruction const* instructions = nullptr;
  int size                                  = 0;
};
struct lazy_scalar {
  __int128_t number = 0;
  bool valid        = false;
};
__device__ __int128_t rescale(__int128_t value, int delta)
{
  for (; delta > 0; --delta)
    value *= 10;
  for (; delta < 0; ++delta)
    value /= 10;
  return value;
}
__device__ bool column_valid(lazy_join_column const& c, int row)
{
  int bit = row + c.offset;
  return !c.mask || ((c.mask[bit / 32] >> (bit % 32)) & 1);
}
__device__ lazy_scalar checked(lazy_scalar value,
                               lazy_join_instruction const& instruction,
                               int* error)
{
  if (value.valid && (value.number < instruction.minimum || value.number > instruction.maximum)) {
    atomicExch(error, 1);
    value.valid = false;
  }
  return value;
}
__device__ lazy_scalar lazy_leaf(lazy_join_instruction const& instruction,
                                 lazy_join_column const* columns,
                                 int row,
                                 int* error)
{
  using op = lazy_join_opcode;
  if (instruction.op == op::literal) return {instruction.number, instruction.valid};
  auto const& c = columns[instruction.column];
  if (!column_valid(c, row) || !instruction.valid) return {};
  int index = row + c.offset;
  if (instruction.op == op::column) {
    __int128_t value = 0;
    switch (static_cast<cudf::type_id>(c.type)) {
      case cudf::type_id::BOOL8: value = static_cast<bool const*>(c.head)[index]; break;
      case cudf::type_id::INT8: value = static_cast<int8_t const*>(c.head)[index]; break;
      case cudf::type_id::INT16: value = static_cast<int16_t const*>(c.head)[index]; break;
      case cudf::type_id::INT32:
      case cudf::type_id::DECIMAL32: value = static_cast<int32_t const*>(c.head)[index]; break;
      case cudf::type_id::INT64:
      case cudf::type_id::DECIMAL64: value = static_cast<int64_t const*>(c.head)[index]; break;
      case cudf::type_id::DECIMAL128: value = static_cast<__int128_t const*>(c.head)[index]; break;
      default: atomicExch(error, 1); return {};
    }
    return checked({rescale(value, instruction.scale - c.scale), true}, instruction, error);
  }
  auto offset = [&](int i) -> int64_t {
    return c.wide_offsets ? static_cast<int64_t const*>(c.offsets)[i]
                          : static_cast<int32_t const*>(c.offsets)[i];
  };
  int64_t first = offset(index), length = offset(index + 1) - first;
  bool equal = instruction.op == op::string_prefix ? length >= instruction.text_size
                                                   : length == instruction.text_size;
  for (int i = 0; equal && i < instruction.text_size; ++i)
    equal = c.chars[first + i] == instruction.text[i];
  return {instruction.op == op::string_not_equal ? !equal : equal, true};
}
__device__ lazy_scalar evaluate_lazy(lazy_device_program program,
                                     lazy_join_column const* columns,
                                     int row,
                                     int* error)
{
  lazy_scalar stack[lazy_join_stack_size];
  int size = 0;
  for (int i = 0; i < program.size; ++i) {
    auto const& instruction = program.instructions[i];
    using op                = lazy_join_opcode;
    switch (instruction.op) {
      case op::column:
      case op::literal:
      case op::string_equal:
      case op::string_not_equal:
      case op::string_prefix: stack[size++] = lazy_leaf(instruction, columns, row, error); break;
      case op::cast:
        stack[size - 1] =
          checked({rescale(stack[size - 1].number, instruction.scale - instruction.lhs_scale),
                   stack[size - 1].valid},
                  instruction,
                  error);
        break;
      case op::logical_not: stack[size - 1].number = !stack[size - 1].number; break;
      default: {
        auto b = stack[--size];
        auto a = stack[size - 1];
        lazy_scalar result;
        if (instruction.op == op::logical_and) {
          if ((a.valid && !a.number) || (b.valid && !b.number))
            result = {0, true};
          else
            result = {1, a.valid && b.valid};
        } else if (instruction.op == op::logical_or) {
          if ((a.valid && a.number) || (b.valid && b.number))
            result = {1, true};
          else
            result = {0, a.valid && b.valid};
        } else if (a.valid && b.valid) {
          if (instruction.op == op::mul)
            result = checked({a.number * b.number, true}, instruction, error);
          else {
            auto lhs = rescale(a.number, instruction.scale - instruction.lhs_scale);
            auto rhs = rescale(b.number, instruction.scale - instruction.rhs_scale);
            if (instruction.op == op::add)
              result = checked({lhs + rhs, true}, instruction, error);
            else if (instruction.op == op::sub)
              result = checked({lhs - rhs, true}, instruction, error);
            else if (instruction.op == op::equal)
              result = {lhs == rhs, true};
            else if (instruction.op == op::not_equal)
              result = {lhs != rhs, true};
          }
        }
        stack[size - 1] = result;
        break;
      }
    }
  }
  return stack[0];
}
struct arguments {
  device_column bk, pk, group;
  device_column values[max_aggregates], predicates[max_aggregates];
  __int128_t* sums[max_aggregates];
  int64_t* valid[max_aggregates];
  bool conditional[max_aggregates], has_predicate[max_aggregates], has_group;
  bool value_is_build[max_aggregates], predicate_is_build[max_aggregates], group_is_build;
  int64_t* matches;
  int aggregates, groups, warps;
  lazy_device_program lazy_values[max_aggregates], lazy_predicates[max_aggregates];
  lazy_join_column const* lazy_columns = nullptr;
  int probe_column_offset              = 0;
  int* expression_error                = nullptr;
  int64_t* expression_evaluations      = nullptr;
  bool lazy                            = false;
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
  int i                         = blockIdx.x * blockDim.x + threadIdx.x;
  bool active                   = i < a.pk.size() && a.pk.is_valid(i);
  int64_t key                   = active ? a.pk.data<int64_t>()[i] : 0;
  int first                     = active ? heads[hash_key(key, mask)] : -1;
  int warp                      = i / 32;
  int64_t predicate_evaluations = 0, value_evaluations = 0;
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
        bool keep         = true;
        if (a.has_predicate[k]) {
          if (a.lazy) {
            auto predicate =
              evaluate_lazy(a.lazy_predicates[k],
                            a.lazy_columns + (a.predicate_is_build[k] ? 0 : a.probe_column_offset),
                            predicate_row,
                            a.expression_error);
            keep = predicate.valid && predicate.number != 0;
            if (a.expression_evaluations) ++predicate_evaluations;
          } else
            keep = a.predicates[k].is_valid(predicate_row) &&
                   a.predicates[k].data<bool>()[predicate_row];
        }
        if (!keep) {
          if (a.conditional[k]) ++valid[k];
          continue;
        }
        int value_row = a.value_is_build[k] ? j : i;
        lazy_scalar value;
        if (a.lazy) {
          value = evaluate_lazy(a.lazy_values[k],
                                a.lazy_columns + (a.value_is_build[k] ? 0 : a.probe_column_offset),
                                value_row,
                                a.expression_error);
          if (a.expression_evaluations) ++value_evaluations;
        } else
          value = {a.values[k].is_valid(value_row) ? a.values[k].data<__int128_t>()[value_row] : 0,
                   a.values[k].is_valid(value_row)};
        if (value.valid) {
          sums[k] += value.number;
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
  if (a.expression_evaluations) {
    for (int d = 16; d; d /= 2) {
      predicate_evaluations += __shfl_down_sync(0xffffffff, predicate_evaluations, d);
      value_evaluations += __shfl_down_sync(0xffffffff, value_evaluations, d);
    }
    if (threadIdx.x % 32 == 0) {
      atomicAdd(reinterpret_cast<unsigned long long*>(a.expression_evaluations),
                predicate_evaluations);
      atomicAdd(reinterpret_cast<unsigned long long*>(a.expression_evaluations + 1),
                value_evaluations);
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
                                   rmm::device_async_resource_ref mr,
                                   lazy_join_inputs const* lazy)
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
  std::vector<rmm::device_uvector<lazy_join_instruction>> programs;
  rmm::device_uvector<lazy_join_column> columns(
    lazy ? lazy->build.num_columns() + lazy->probe.num_columns() : 0, stream, mr);
  if (lazy) {
    auto describe = [&](cudf::column_view c) {
      lazy_join_column description;
      description.head   = c.head();
      description.mask   = c.null_mask();
      description.offset = c.offset();
      description.type   = static_cast<int>(c.type().id());
      if (c.type().id() == cudf::type_id::DECIMAL32 || c.type().id() == cudf::type_id::DECIMAL64 ||
          c.type().id() == cudf::type_id::DECIMAL128)
        description.scale = -static_cast<int>(c.type().scale());
      if (c.type().id() == cudf::type_id::STRING && c.num_children()) {
        cudf::strings_column_view strings(c);
        description.offsets      = strings.offsets().head();
        description.chars        = strings.chars_begin(stream);
        description.wide_offsets = strings.offsets().type().id() == cudf::type_id::INT64;
      }
      out.host_columns.push_back(description);
    };
    for (auto c : lazy->build)
      describe(c);
    for (auto c : lazy->probe)
      describe(c);
    CUDF_CUDA_TRY(cudaMemcpyAsync(columns.data(),
                                  out.host_columns.data(),
                                  columns.size() * sizeof(lazy_join_column),
                                  cudaMemcpyHostToDevice,
                                  stream.get()));
    a.lazy                = true;
    a.lazy_columns        = columns.data();
    a.probe_column_offset = lazy->build.num_columns();
    out.errors            = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, 1, cudf::mask_state::UNALLOCATED, stream, mr);
    a.expression_error = out.errors->mutable_view().data<int32_t>();
    CUDF_CUDA_TRY(cudaMemsetAsync(a.expression_error, 0, sizeof(int32_t), stream.get()));
    if (lazy->audit) {
      out.expression_evaluations = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT64}, 2, cudf::mask_state::UNALLOCATED, stream, mr);
      a.expression_evaluations = out.expression_evaluations->mutable_view().data<int64_t>();
      CUDF_CUDA_TRY(
        cudaMemsetAsync(a.expression_evaluations, 0, 2 * sizeof(int64_t), stream.get()));
    }
  }
  auto upload = [&](lazy_join_expression const& program) {
    programs.emplace_back(program.instructions.size(), stream, mr);
    if (!program.instructions.empty())
      CUDF_CUDA_TRY(cudaMemcpyAsync(programs.back().data(),
                                    program.instructions.data(),
                                    program.instructions.size() * sizeof(lazy_join_instruction),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));
    return lazy_device_program{programs.back().data(),
                               static_cast<int>(program.instructions.size())};
  };
  for (size_t k = 0; k < values.size(); k++) {
    out.sums.push_back(cudf::make_fixed_width_column(
      values[k].type(), groups * warps, cudf::mask_state::UNALLOCATED, stream, mr));
    out.valid_counts.push_back(cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                                         groups * warps,
                                                         cudf::mask_state::UNALLOCATED,
                                                         stream,
                                                         mr));
    a.values[k]        = view(values[k]);
    a.has_predicate[k] = lazy ? !lazy->predicates[k].instructions.empty()
                              : predicates[k].type().id() != cudf::type_id::EMPTY;
    if (lazy) {
      a.lazy_values[k]     = upload(lazy->values[k]);
      a.lazy_predicates[k] = upload(lazy->predicates[k]);
    }
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
