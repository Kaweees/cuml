/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuml/fil/detail/device_initialization/gpu.cuh>
#include <cuml/fil/detail/infer/gpu.cuh>
#include <cuml/fil/detail/specializations/device_initialization_macros.hpp>
#include <cuml/fil/detail/specializations/infer_macros.hpp>
#pragma GCC visibility push(default)
namespace ML {
namespace fil {
namespace detail {
namespace inference {
CUML_FIL_INFER_ALL(template, raft_proto::device_type::gpu, 2)
}
namespace device_initialization {
CUML_FIL_INITIALIZE_DEVICE(template, 2)
}
}  // namespace detail
}  // namespace fil
#pragma GCC visibility pop
}  // namespace ML
