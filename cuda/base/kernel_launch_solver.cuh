// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_SOLVER_HPP_
#error \
    "This file can only be used from inside common/unified/base/kernel_launch_solver.hpp"
#endif


#include "common/cuda_hip/base/runtime.hpp"


namespace gko {
namespace kernels {
namespace cuda {


#include "common/cuda_hip/base/kernel_launch_solver.hpp.inc"


}  // namespace cuda
}  // namespace kernels
}  // namespace gko
