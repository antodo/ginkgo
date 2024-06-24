// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/par_ilu_kernels.hpp"


#include <ginkgo/core/matrix/coo.hpp>


#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/memory.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The parallel ilu factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ilu_factorization {


constexpr int default_block_size{512};


#include "common/cuda_hip/factorization/par_ilu_kernels.hpp.inc"


}  // namespace par_ilu_factorization
}  // namespace hip
}  // namespace kernels
}  // namespace gko
