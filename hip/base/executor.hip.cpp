/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/base/executor.hpp>


#include <iostream>


#include <hip/hip_runtime.h>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/device.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


#include "hip/base/config.hip.hpp"
#include "hip/base/hipblas_bindings.hip.hpp"
#include "hip/base/hipsparse_bindings.hip.hpp"
#include "hip/base/scoped_device_id.hip.hpp"


namespace gko {


#include "common/cuda_hip/base/executor.hpp.inc"


std::shared_ptr<HipExecutor> HipExecutor::create(
    int device_id, std::shared_ptr<Executor> master, bool device_reset,
    allocation_mode alloc_mode, hipStream_t stream)
{
    return std::shared_ptr<HipExecutor>(
        new HipExecutor(device_id, std::move(master),
                        std::make_shared<HipAllocator>(), stream));
}


std::shared_ptr<HipExecutor> HipExecutor::create(
    int device_id, std::shared_ptr<Executor> master,
    std::shared_ptr<HipAllocatorBase> alloc, hipStream_t stream)
{
    return std::shared_ptr<HipExecutor>(new HipExecutor(
        device_id, std::move(master), std::move(alloc), stream));
}


void HipExecutor::populate_exec_info(const machine_topology* mach_topo)
{
    if (this->get_device_id() < this->get_num_devices() &&
        this->get_device_id() >= 0) {
        detail::hip_scoped_device_id_guard g(this->get_device_id());
        GKO_ASSERT_NO_HIP_ERRORS(
            hipDeviceGetPCIBusId(&(this->get_exec_info().pci_bus_id.front()),
                                 13, this->get_device_id()));

        auto hip_hwloc_obj =
            mach_topo->get_pci_device(this->get_exec_info().pci_bus_id);
        if (hip_hwloc_obj) {
            this->get_exec_info().numa_node = hip_hwloc_obj->closest_numa;
            this->get_exec_info().closest_pu_ids =
                hip_hwloc_obj->closest_pu_ids;
        }
    }
}


void OmpExecutor::raw_copy_to(const HipExecutor* dest, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
{
    if (num_bytes > 0) {
        detail::hip_scoped_device_id_guard g(dest->get_device_id());
        GKO_ASSERT_NO_HIP_ERRORS(hipMemcpyAsync(dest_ptr, src_ptr, num_bytes,
                                                hipMemcpyHostToDevice,
                                                dest->get_stream()));
        dest->synchronize();
    }
}


void HipExecutor::raw_free(void* ptr) const noexcept
{
    detail::hip_scoped_device_id_guard g(this->get_device_id());
    alloc_->deallocate(ptr);
}


void* HipExecutor::raw_alloc(size_type num_bytes) const
{
    detail::hip_scoped_device_id_guard g(this->get_device_id());
    return alloc_->allocate(num_bytes);
}


void HipExecutor::raw_copy_to(const OmpExecutor*, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
{
    if (num_bytes > 0) {
        detail::hip_scoped_device_id_guard g(this->get_device_id());
        GKO_ASSERT_NO_HIP_ERRORS(hipMemcpyAsync(dest_ptr, src_ptr, num_bytes,
                                                hipMemcpyDeviceToHost,
                                                this->get_stream()));
        this->synchronize();
    }
}


void HipExecutor::raw_copy_to(const CudaExecutor* dest, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
{
#if GINKGO_HIP_PLATFORM_NVCC == 1
    if (num_bytes > 0) {
        detail::hip_scoped_device_id_guard g(this->get_device_id());
        GKO_ASSERT_NO_HIP_ERRORS(hipMemcpyPeerAsync(
            dest_ptr, dest->get_device_id(), src_ptr, this->get_device_id(),
            num_bytes, this->get_stream()));
        this->synchronize();
    }
#else
    GKO_NOT_SUPPORTED(dest);
#endif
}


void HipExecutor::raw_copy_to(const DpcppExecutor* dest, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
{
    GKO_NOT_SUPPORTED(dest);
}


void HipExecutor::raw_copy_to(const HipExecutor* dest, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
{
    if (num_bytes > 0) {
        detail::hip_scoped_device_id_guard g(this->get_device_id());
        GKO_ASSERT_NO_HIP_ERRORS(hipMemcpyPeerAsync(
            dest_ptr, dest->get_device_id(), src_ptr, this->get_device_id(),
            num_bytes, this->get_stream()));
        this->synchronize();
    }
}


void HipExecutor::synchronize() const
{
    detail::hip_scoped_device_id_guard g(this->get_device_id());
    GKO_ASSERT_NO_HIP_ERRORS(hipStreamSynchronize(this->get_stream()));
}


scoped_device_id_guard HipExecutor::get_scoped_device_id_guard() const
{
    return {this, this->get_device_id()};
}


int HipExecutor::get_num_devices()
{
    int deviceCount = 0;
    auto error_code = hipGetDeviceCount(&deviceCount);
    if (error_code == hipErrorNoDevice) {
        return 0;
    }
    GKO_ASSERT_NO_HIP_ERRORS(error_code);
    return deviceCount;
}


void HipExecutor::set_gpu_property()
{
    if (this->get_device_id() < this->get_num_devices() &&
        this->get_device_id() >= 0) {
        detail::hip_scoped_device_id_guard g(this->get_device_id());
        GKO_ASSERT_NO_HIP_ERRORS(hipDeviceGetAttribute(
            &this->get_exec_info().num_computing_units,
            hipDeviceAttributeMultiprocessorCount, this->get_device_id()));
        GKO_ASSERT_NO_HIP_ERRORS(hipDeviceGetAttribute(
            &this->get_exec_info().major,
            hipDeviceAttributeComputeCapabilityMajor, this->get_device_id()));
        GKO_ASSERT_NO_HIP_ERRORS(hipDeviceGetAttribute(
            &this->get_exec_info().minor,
            hipDeviceAttributeComputeCapabilityMinor, this->get_device_id()));
        auto max_threads_per_block = 0;
        GKO_ASSERT_NO_HIP_ERRORS(hipDeviceGetAttribute(
            &max_threads_per_block, hipDeviceAttributeMaxThreadsPerBlock,
            this->get_device_id()));
        this->get_exec_info().max_workitem_sizes.push_back(
            max_threads_per_block);
        std::vector<int> max_threads_per_block_dim(3, 0);
        GKO_ASSERT_NO_HIP_ERRORS(hipDeviceGetAttribute(
            &max_threads_per_block_dim[0], hipDeviceAttributeMaxBlockDimX,
            this->get_device_id()));
        GKO_ASSERT_NO_HIP_ERRORS(hipDeviceGetAttribute(
            &max_threads_per_block_dim[1], hipDeviceAttributeMaxBlockDimY,
            this->get_device_id()));
        GKO_ASSERT_NO_HIP_ERRORS(hipDeviceGetAttribute(
            &max_threads_per_block_dim[2], hipDeviceAttributeMaxBlockDimZ,
            this->get_device_id()));
        this->get_exec_info().max_workgroup_size = max_threads_per_block;
        this->get_exec_info().max_workitem_sizes = max_threads_per_block_dim;
#if GINKGO_HIP_PLATFORM_NVCC
        this->get_exec_info().num_pu_per_cu =
            convert_sm_ver_to_cores(this->get_exec_info().major,
                                    this->get_exec_info().minor) /
            kernels::hip::config::warp_size;
#else
        // In GCN (Graphics Core Next), each multiprocessor has 4 SIMD
        // Reference: https://en.wikipedia.org/wiki/Graphics_Core_Next
        this->get_exec_info().num_pu_per_cu = 4;
#endif  // GINKGO_HIP_PLATFORM_NVCC
        this->get_exec_info().max_subgroup_size =
            kernels::hip::config::warp_size;
    }
}


void HipExecutor::init_handles()
{
    if (this->get_device_id() < this->get_num_devices() &&
        this->get_device_id() >= 0) {
        const auto id = this->get_device_id();
        detail::hip_scoped_device_id_guard g(id);
        this->hipblas_handle_ = handle_manager<hipblasContext>(
            kernels::hip::hipblas::init(this->get_stream()),
            [id](hipblasContext* handle) {
                detail::hip_scoped_device_id_guard g(id);
                kernels::hip::hipblas::destroy_hipblas_handle(handle);
            });
        this->hipsparse_handle_ = handle_manager<hipsparseContext>(
            kernels::hip::hipsparse::init(this->get_stream()),
            [id](hipsparseContext* handle) {
                detail::hip_scoped_device_id_guard g(id);
                kernels::hip::hipsparse::destroy_hipsparse_handle(handle);
            });
    }
}


}  // namespace gko
