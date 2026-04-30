/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuml/fil/detail/infer/cpu.hpp>
#include <cuml/fil/detail/specializations/infer_macros.hpp>
#pragma GCC visibility push(default)
namespace ML {
namespace fil {
namespace detail {
namespace inference {
CUML_FIL_INFER_ALL(template, raft_proto::device_type::cpu, 10)
}
}  // namespace detail
}  // namespace fil
#pragma GCC visibility pop
}  // namespace ML
