// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
// Capability check, not an engine performance benchmark.
#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/join/join.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include <cassert>
#include <iostream>

__global__ void consume(cudf::table_device_view left,
                        cudf::table_device_view right,
                        cudf::ast::detail::expression_device_view plan,
                        int const* li,
                        int const* ri,
                        int matches,
                        int scratch_per_thread,
                        long long* totals)
{
  extern __shared__ unsigned char scratch[];
  using storage   = cudf::ast::detail::IntermediateDataType<true>;
  auto* temporary = reinterpret_cast<storage*>(scratch) + threadIdx.x * scratch_per_thread;
  cudf::ast::detail::expression_evaluator<true> eval(left, right, plan);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= matches) return;
  cudf::ast::detail::value_expression_result<int64_t, true> value;
  eval.evaluate(value, li[i], ri[i], 0, temporary);
  atomicAdd(reinterpret_cast<unsigned long long*>(totals + 2), 1ULL);
  if (value.is_valid()) {
    atomicAdd(reinterpret_cast<unsigned long long*>(totals),
              static_cast<unsigned long long>(value.value()));
    atomicAdd(reinterpret_cast<unsigned long long*>(totals + 1), 1ULL);
  }
}
int main()
{
  auto stream = cudf::get_default_stream();
  auto column = [&](auto const& data) {
    auto result = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64}, 3, cudf::mask_state::UNALLOCATED, stream);
    CUDF_CUDA_TRY(cudaMemcpy(
      result->mutable_view().data<int64_t>(), data, 3 * sizeof(int64_t), cudaMemcpyHostToDevice));
    return result;
  };
  int64_t lk[] = {1, 1, 3}, rk[] = {1, 1, 2}, lv[] = {2, 4, 9}, rv[] = {10, 20, 99};
  auto left_key = column(lk), right_key = column(rk), left_value = column(lv),
       right_value        = column(rv);
  auto mask               = cudf::create_null_mask(3, cudf::mask_state::ALL_VALID, stream);
  cudf::bitmask_type bits = 5;
  CUDF_CUDA_TRY(cudaMemcpy(mask.data(), &bits, sizeof(bits), cudaMemcpyHostToDevice));
  left_value->set_null_mask(std::move(mask), 1);
  cudf::table_view left{{left_key->view(), left_value->view()}},
    right{{right_key->view(), right_value->view()}};
  auto maps = cudf::inner_join(cudf::table_view{{left_key->view()}},
                               cudf::table_view{{right_key->view()}},
                               cudf::null_equality::UNEQUAL,
                               stream);
  cudf::ast::tree tree;
  auto& a    = tree.emplace<cudf::ast::column_reference>(1, cudf::ast::table_reference::LEFT);
  auto& b    = tree.emplace<cudf::ast::column_reference>(1, cudf::ast::table_reference::RIGHT);
  auto& add  = tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::ADD, a, b);
  auto& root = tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::MUL, add, a);
  cudf::ast::detail::expression_parser parser(
    root, left, std::cref(right), true, stream, cudf::get_current_device_resource_ref());
  auto dl = cudf::table_device_view::create(left, stream),
       dr = cudf::table_device_view::create(right, stream);
  rmm::device_uvector<long long> result(3, stream);
  CUDF_CUDA_TRY(cudaMemsetAsync(result.data(), 0, 3 * sizeof(long long), stream.get()));
  using storage = cudf::ast::detail::IntermediateDataType<true>;
  assert(parser.shmem_per_thread % sizeof(storage) == 0);
  consume<<<1, 128, 128 * parser.shmem_per_thread, stream.get()>>>(
    *dl,
    *dr,
    parser.device_expression_data,
    maps.first->data(),
    maps.second->data(),
    maps.first->size(),
    parser.shmem_per_thread / sizeof(storage),
    result.data());
  CUDF_CUDA_TRY(cudaGetLastError());
  long long host[3];
  CUDF_CUDA_TRY(cudaMemcpy(host, result.data(), sizeof(host), cudaMemcpyDeviceToHost));
  assert(host[0] == 68 && host[1] == 2 && host[2] == 4);
  std::cout << "cuDF two-table row evaluator + ordinary join maps: sum=" << host[0]
            << " valid=" << host[1] << " pairs=" << host[2]
            << "; no gathered payload or expression-result columns\n";
}
