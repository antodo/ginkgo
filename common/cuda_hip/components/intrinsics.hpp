// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_COMPONENTS_INTRINSICS_HPP_
#define GKO_COMMON_CUDA_HIP_COMPONENTS_INTRINSICS_HPP_


#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {


/**
 * @internal
 * Returns the number of set bits in the given mask.
 */
__forceinline__ __device__ int popcnt(uint32 mask) { return __popc(mask); }

/** @copydoc popcnt */
__forceinline__ __device__ int popcnt(uint64 mask) { return __popcll(mask); }


/**
 * @internal
 * Returns the (1-based!) index of the first set bit in the given mask,
 * starting from the least significant bit.
 */
__forceinline__ __device__ int ffs(uint32 mask) { return __ffs(mask); }

/** @copydoc ffs */
__forceinline__ __device__ int ffs(uint64 mask)
{
    // the cast is necessary, as the overloads defined by HIP are ambiguous
    return __ffsll(static_cast<unsigned long long int>(mask));
}


/**
 * @internal
 * Returns the number of zero bits before the first set bit in the given mask,
 * starting from the most significant bit.
 */
__forceinline__ __device__ int clz(uint32 mask) { return __clz(mask); }

/** @copydoc clz */
__forceinline__ __device__ int clz(uint64 mask) { return __clzll(mask); }


}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif  // GKO_COMMON_CUDA_HIP_COMPONENTS_INTRINSICS_HPP_
