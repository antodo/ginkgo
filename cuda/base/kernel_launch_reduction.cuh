// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_REDUCTION_HPP_
#error \
    "This file can only be used from inside common/unified/base/kernel_launch_reduction.hpp"
#endif


#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/components/reduction.cuh"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {


#include "common/cuda_hip/base/kernel_launch_reduction.hpp.inc"


}  // namespace cuda
}  // namespace kernels
}  // namespace gko
