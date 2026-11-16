// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "expression/ast/node.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <optional>
#include <vector>

namespace sirius::op {
// A deliberately bounded scalar IR, evaluated only after equality matching.
// Unsupported ASTs retain the ordinary engine plan. No row-level output is emitted.
enum class lazy_join_opcode : int {
  column,
  literal,
  cast,
  add,
  sub,
  mul,
  equal,
  not_equal,
  logical_and,
  logical_or,
  string_equal,
  string_not_equal,
  string_prefix,
  logical_not
};
inline constexpr int lazy_join_stack_size       = 16;
inline constexpr int lazy_join_max_instructions = 32;
inline constexpr int lazy_join_literal_bytes    = 64;
struct lazy_join_instruction {
  lazy_join_opcode op{};
  int column = 0, scale = 0, lhs_scale = 0, rhs_scale = 0, text_size = 0;
  __int128_t number = 0, minimum = 0, maximum = 0;
  bool valid = true;
  char text[lazy_join_literal_bytes]{};
};
struct lazy_join_expression {
  std::vector<lazy_join_instruction> instructions;
};
std::optional<lazy_join_expression> compile_lazy_join_expression(ast::node const& expression);
// Host-owned descriptors are retained until stream work completes; they contain
// device pointers, never pointers to a host-owned column_device_view.
struct lazy_join_column {
  void const* head               = nullptr;
  cudf::bitmask_type const* mask = nullptr;
  void const* offsets            = nullptr;
  char const* chars              = nullptr;
  int offset = 0, type = 0, scale = 0;
  bool wide_offsets = false;
};
struct lazy_join_inputs {
  std::vector<lazy_join_expression> values, predicates;
  cudf::table_view build, probe;
  bool audit = false;
};
}  // namespace sirius::op
