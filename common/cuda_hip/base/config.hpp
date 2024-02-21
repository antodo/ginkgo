// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_BASE_CONFIG_HPP_
#define GKO_COMMON_CUDA_HIP_BASE_CONFIG_HPP_


#ifdef GKO_COMPILING_HIP
#include "hip/base/config.hip.hpp"
#else  // GKO_COMPILING_CUDA
#include "cuda/base/config.hpp"
#endif


#endif  // GKO_COMMON_CUDA_HIP_BASE_CONFIG_HPP_
