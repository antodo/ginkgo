// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/batch_cg_kernels.hpp"

#include <thrust/functional.h>
#include <thrust/transform.h>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>

#include "common/cuda_hip/base/batch_multi_vector_kernels.hpp"
#include "common/cuda_hip/base/batch_struct.hpp"
#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/thrust.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/reduction.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "common/cuda_hip/components/uninitialized_array.hpp"
#include "common/cuda_hip/matrix/batch_csr_kernels.hpp"
#include "common/cuda_hip/matrix/batch_dense_kernels.hpp"
#include "common/cuda_hip/matrix/batch_ell_kernels.hpp"
#include "common/cuda_hip/matrix/batch_struct.hpp"
#include "common/cuda_hip/solver/batch_cg_kernels.hpp"
#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "core/solver/batch_dispatch.hpp"


namespace gko {
namespace kernels {
namespace hip {
namespace batch_cg {


template <typename BatchMatrixType>
int get_num_threads_per_block(std::shared_ptr<const DefaultExecutor> exec,
                              const int num_rows)
{
    int num_warps = std::max(num_rows / 4, 2);
    constexpr int warp_sz = static_cast<int>(config::warp_size);
    const int min_block_size = 2 * warp_sz;
    const int device_max_threads =
        ((std::max(num_rows, min_block_size)) / warp_sz) * warp_sz;
    // This value has been taken from ROCm docs. This is the number of registers
    // that maximizes the occupancy on an AMD GPU (MI200). HIP does not have an
    // API to query the number of registers a function uses.
    const int num_regs_used_per_thread = 64;
    int max_regs_blk = 0;
    GKO_ASSERT_NO_HIP_ERRORS(hipDeviceGetAttribute(
        &max_regs_blk, hipDeviceAttributeMaxRegistersPerBlock,
        exec->get_device_id()));
    int max_threads_regs = (max_regs_blk / num_regs_used_per_thread);
    max_threads_regs = (max_threads_regs / warp_sz) * warp_sz;
    int max_threads = std::min(max_threads_regs, device_max_threads);
    max_threads = max_threads <= 1024 ? max_threads : 1024;
    return std::max(std::min(num_warps * warp_sz, max_threads), min_block_size);
}


template <typename T>
using settings = gko::kernels::batch_cg::settings<T>;


template <typename HipValueType>
class kernel_caller {
public:
    using value_type = HipValueType;

    kernel_caller(std::shared_ptr<const DefaultExecutor> exec,
                  const settings<remove_complex<value_type>> settings)
        : exec_{exec}, settings_{settings}
    {}

    template <typename StopType, const int n_shared,
              const bool prec_shared_bool, typename PrecType, typename LogType,
              typename BatchMatrixType>
    void launch_apply_kernel(
        const gko::kernels::batch_cg::storage_config& sconf, LogType& logger,
        PrecType& prec, const BatchMatrixType& mat,
        const value_type* const __restrict__ b_values,
        value_type* const __restrict__ x_values,
        value_type* const __restrict__ workspace_data, const int& block_size,
        const size_t& shared_size) const
    {
        batch_single_kernels::apply_kernel<StopType, n_shared, prec_shared_bool>
            <<<mat.num_batch_items, block_size, shared_size,
               exec_->get_stream()>>>(sconf, settings_.max_iterations,
                                      settings_.residual_tol, logger, prec, mat,
                                      b_values, x_values, workspace_data);
    }


    template <typename BatchMatrixType, typename PrecType, typename StopType,
              typename LogType>
    void call_kernel(
        LogType logger, const BatchMatrixType& mat, PrecType prec,
        const gko::batch::multi_vector::uniform_batch<const value_type>& b,
        const gko::batch::multi_vector::uniform_batch<value_type>& x) const
    {
        using real_type = gko::remove_complex<value_type>;
        const size_type num_batch_items = mat.num_batch_items;
        constexpr int align_multiple = 8;
        const int padded_num_rows =
            ceildiv(mat.num_rows, align_multiple) * align_multiple;
        int shmem_per_blk = 0;
        GKO_ASSERT_NO_HIP_ERRORS(hipDeviceGetAttribute(
            &shmem_per_blk, hipDeviceAttributeMaxSharedMemoryPerBlock,
            exec_->get_device_id()));
        const int block_size =
            get_num_threads_per_block<BatchMatrixType>(exec_, mat.num_rows);
        bool is_block_size_aligned = block_size % config::warp_size == 0;
        GKO_ASSERT(block_size >= 2 * config::warp_size);
        GKO_ASSERT(is_block_size_aligned);

        // Returns amount required in bytes
        const size_t prec_size = PrecType::dynamic_work_size(
            padded_num_rows, mat.get_single_item_num_nnz());
        const auto sconf =
            gko::kernels::batch_cg::compute_shared_storage<PrecType,
                                                           value_type>(
                shmem_per_blk, padded_num_rows, mat.get_single_item_num_nnz(),
                b.num_rhs);
        const size_t shared_size =
            sconf.n_shared * padded_num_rows * sizeof(value_type) +
            (sconf.prec_shared ? prec_size : 0);
        auto workspace = gko::array<value_type>(
            exec_,
            sconf.gmem_stride_bytes * num_batch_items / sizeof(value_type));
        bool is_stride_aligned =
            sconf.gmem_stride_bytes % sizeof(value_type) == 0;
        GKO_ASSERT(is_stride_aligned);

        value_type* const workspace_data = workspace.get_data();

        // Only instantiate when full optimizations has been enabled. Otherwise,
        // just use the default one with no shared memory.
        // Template parameters launch_apply_kernel<StopType, n_shared,
        // prec_shared)
        // if (sconf.prec_shared) {
        //     launch_apply_kernel<StopType, 5, true>(
        //         sconf, logger, prec, mat, b.values, x.values, workspace_data,
        //         block_size, shared_size);
        // } else {
        //     switch (sconf.n_shared) {
        //     case 0:
        launch_apply_kernel<StopType, 0, false>(
            sconf, logger, prec, mat, b.values, x.values, workspace_data,
            block_size, shared_size);
        //         break;
        //     case 1:
        //         launch_apply_kernel<StopType, 1, false>(
        //             sconf, logger, prec, mat, b.values, x.values,
        //             workspace_data, block_size, shared_size);
        //         break;
        //     case 2:
        //         launch_apply_kernel<StopType, 2, false>(
        //             sconf, logger, prec, mat, b.values, x.values,
        //             workspace_data, block_size, shared_size);
        //         break;
        //     case 3:
        //         launch_apply_kernel<StopType, 3, false>(
        //             sconf, logger, prec, mat, b.values, x.values,
        //             workspace_data, block_size, shared_size);
        //         break;
        //     case 4:
        //         launch_apply_kernel<StopType, 4, false>(
        //             sconf, logger, prec, mat, b.values, x.values,
        //             workspace_data, block_size, shared_size);
        //         break;
        //     case 5:
        //         launch_apply_kernel<StopType, 5, false>(
        //             sconf, logger, prec, mat, b.values, x.values,
        //             workspace_data, block_size, shared_size);
        //         break;
        //     default:
        //         GKO_NOT_IMPLEMENTED;
        //     }
        // }
    }

private:
    std::shared_ptr<const DefaultExecutor> exec_;
    const settings<remove_complex<value_type>> settings_;
};


template <typename ValueType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           const settings<remove_complex<ValueType>>& settings,
           const batch::BatchLinOp* const mat,
           const batch::BatchLinOp* const precon,
           const batch::MultiVector<ValueType>* const b,
           batch::MultiVector<ValueType>* const x,
           batch::log::detail::log_data<remove_complex<ValueType>>& logdata)
{
    using hip_value_type = hip_type<ValueType>;
    auto dispatcher = batch::solver::create_dispatcher<ValueType>(
        kernel_caller<hip_value_type>(exec, settings), settings, mat, precon);
    dispatcher.apply(b, x, logdata);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_CG_APPLY_KERNEL);


}  // namespace batch_cg
}  // namespace hip
}  // namespace kernels
}  // namespace gko
