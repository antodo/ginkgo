// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/reorder/rcm_kernels.hpp"

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "common/cuda_hip/components/memory.hpp"
#include "core/base/array_access.hpp"
#include "cuda/base/thrust.cuh"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The reordering namespace.
 *
 * @ingroup reorder
 */
namespace rcm {


constexpr int default_block_size = 512;


#include "common/cuda_hip/reorder/rcm_kernels.hpp.inc"


}  // namespace rcm
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
