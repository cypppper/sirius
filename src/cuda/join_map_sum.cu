// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "expression_evaluator/gpu_expression_translator_internal.hpp"
#include "log/logging.hpp"
#include "op/aggregate/join_map_sum.hpp"

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/error.hpp>

#include <cuda_runtime.h>
namespace sirius::op {
namespace {
__device__ __int128_t sum_warp(__int128_t value)
{
  for (int d = 16; d; d /= 2) {
    auto lo = __shfl_down_sync(0xffffffff, static_cast<uint64_t>(value), d);
    auto hi = __shfl_down_sync(0xffffffff, static_cast<uint64_t>(value >> 64), d);
    value += static_cast<__int128_t>((static_cast<__uint128_t>(hi) << 64) | lo);
  }
  return value;
}
template <typename Rep>
__global__ void sum_pairs(cudf::table_device_view left,
                          cudf::table_device_view right,
                          cudf::ast::detail::expression_device_view plan,
                          cudf::size_type const* li,
                          cudf::size_type const* ri,
                          int pairs,
                          int scratch_per_thread,
                          __int128_t* sums,
                          cudf::bitmask_type* mask)
{
  extern __shared__ unsigned char scratch[];
  using storage   = cudf::ast::detail::IntermediateDataType<true>;
  auto* temporary = reinterpret_cast<storage*>(scratch) + threadIdx.x * scratch_per_thread;
  int i           = blockIdx.x * blockDim.x + threadIdx.x;
  cudf::ast::detail::expression_evaluator<true> eval(left, right, plan);
  cudf::ast::detail::value_expression_result<Rep, true> value;
  bool valid     = false;
  __int128_t sum = 0;
  if (i < pairs) {
    eval.evaluate(value, li[i], ri[i], 0, temporary);
    valid = value.is_valid();
    if (valid) sum = value.value();
  }
  unsigned valid_lanes = __ballot_sync(0xffffffff, valid);
  sum                  = sum_warp(sum);
  if (threadIdx.x % 32 == 0) {
    int row   = i / 32;
    sums[row] = sum;
    if (valid_lanes) atomicOr(mask + row / 32, 1u << (row % 32));
  }
}
}  // namespace
join_map_consumer make_join_map_sum_consumer(std::vector<join_map_sum_argument> arguments)
{
  auto owned = std::make_shared<std::vector<join_map_sum_argument>>(std::move(arguments));
  return [owned](cudf::table_view left,
                 cudf::table_view right,
                 rmm::device_uvector<cudf::size_type> const& li,
                 rmm::device_uvector<cudf::size_type> const& ri,
                 ::cuda::stream_ref stream,
                 rmm::device_async_resource_ref mr) {
    auto dl = cudf::table_device_view::create(left, stream);
    auto dr = cudf::table_device_view::create(right, stream);
    std::vector<std::unique_ptr<cudf::column>> outputs;
    int blocks = std::max(1, static_cast<int>((li.size() + 127) / 128));
    for (auto const& argument : *owned) {
      gpu_expression_translator translator(stream, mr);
      auto expression = translator.translate_expression(
        *argument.expression,
        argument.from_right ? cudf::ast::table_reference::RIGHT : cudf::ast::table_reference::LEFT);
      if (!expression) throw std::runtime_error("join-map SUM: expression translation changed");
      // A root reference also needs an operation to populate the scalar result.
      auto& root = expression->tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::IDENTITY,
                                                                  expression->back());
      cudf::ast::detail::expression_parser parser(root, left, std::cref(right), true, stream, mr);
      auto output = cudf::make_fixed_width_column(
        argument.output_type, blocks * 4, cudf::mask_state::ALL_NULL, stream, mr);
      using storage = cudf::ast::detail::IntermediateDataType<true>;
      if (parser.shmem_per_thread % sizeof(storage) || 128 * parser.shmem_per_thread > 48 * 1024)
        throw std::runtime_error("join-map SUM: expression scratch exceeds experimental bound");
      auto launch = [&]<typename Rep>() {
        sum_pairs<Rep><<<blocks, 128, 128 * parser.shmem_per_thread, stream.get()>>>(
          *dl,
          *dr,
          parser.device_expression_data,
          li.data(),
          ri.data(),
          li.size(),
          parser.shmem_per_thread / sizeof(storage),
          output->mutable_view().data<__int128_t>(),
          output->mutable_view().null_mask());
      };
      switch (parser.output_type().id()) {
        case cudf::type_id::INT8: launch.template operator()<int8_t>(); break;
        case cudf::type_id::INT16: launch.template operator()<int16_t>(); break;
        case cudf::type_id::INT32:
        case cudf::type_id::DECIMAL32: launch.template operator()<int32_t>(); break;
        case cudf::type_id::INT64:
        case cudf::type_id::DECIMAL64: launch.template operator()<int64_t>(); break;
        default: throw std::runtime_error("join-map SUM: unsupported scalar output");
      }
      CUDF_CUDA_TRY(cudaGetLastError());
      // Query stream orders kernel completion before these AST/literal owners are released.
      output->set_null_count(
        cudf::null_count(output->view().null_mask(), 0, output->size(), stream));
      outputs.push_back(std::move(output));
    }
    SIRIUS_LOG_INFO("[JOIN_MAP_SUM] pairs={} partial_rows={} sums={} shared_cudf_ast=true",
                    li.size(),
                    blocks * 4,
                    owned->size());
    return std::make_unique<cudf::table>(std::move(outputs));
  };
}
}  // namespace sirius::op
