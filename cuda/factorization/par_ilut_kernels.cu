/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include "core/factorization/par_ilut_kernels.hpp"


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/pointer_mode_guard.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The parallel ilut factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ilut_factorization {


template <typename ValueType, typename IndexType>
void threshold_filter(std::shared_ptr<const CudaExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *a,
                      IndexType target_size,
                      Array<IndexType> &new_row_ptrs_array,
                      Array<IndexType> &new_col_idxs_array,
                      Array<ValueType> &new_vals_array) GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_THRESHOLD_FILTER_KERNEL);


template <typename ValueType, typename IndexType>
void spgeam(std::shared_ptr<const CudaExecutor> exec,
            const matrix::Dense<ValueType> *alpha,
            const matrix::Csr<ValueType, IndexType> *a,
            const matrix::Dense<ValueType> *beta,
            const matrix::Csr<ValueType, IndexType> *b,
            Array<IndexType> &c_row_ptrs_array,
            Array<IndexType> &c_col_idxs_array, Array<ValueType> &c_vals_array)
{
    if (cusparse::is_supported<ValueType, IndexType>::value) {
        auto handle = exec->get_cusparse_handle();
        cusparse::pointer_mode_guard pm_guard(handle);
        auto a_descr = cusparse::create_mat_descr();
        auto b_descr = cusparse::create_mat_descr();
        auto c_descr = cusparse::create_mat_descr();

        ValueType valpha{};
        exec->get_master()->copy_from(exec.get(), 1, alpha->get_const_values(),
                                      &valpha);
        auto a_nnz = IndexType(a->get_num_stored_elements());
        auto a_vals = a->get_const_values();
        auto a_row_ptrs = a->get_const_row_ptrs();
        auto a_col_idxs = a->get_const_col_idxs();
        ValueType vbeta{};
        exec->get_master()->copy_from(exec.get(), 1, beta->get_const_values(),
                                      &vbeta);
        auto b_nnz = IndexType(b->get_num_stored_elements());
        auto b_vals = b->get_const_values();
        auto b_row_ptrs = b->get_const_row_ptrs();
        auto b_col_idxs = b->get_const_col_idxs();
        // don't know why cuSPARSE needs them, as they are invalid anyways
        auto c_nnz = IndexType{};
        auto c_vals = c_vals_array.get_data();
        auto c_row_ptrs = c_row_ptrs_array.get_data();
        auto c_col_idxs = c_col_idxs_array.get_data();
        auto m = IndexType(a->get_size()[0]);
        auto n = IndexType(a->get_size()[1]);

        // allocate buffer
        size_type buffer_size{};
        cusparse::spgeam_buffer_size(
            handle, m, n, &valpha, a_descr, a_nnz, a_vals, a_row_ptrs,
            a_col_idxs, &vbeta, b_descr, b_nnz, b_vals, b_row_ptrs, b_col_idxs,
            c_descr, c_vals, c_row_ptrs, c_col_idxs, buffer_size);
        Array<char> buffer_array(exec, buffer_size);
        auto buffer = buffer_array.get_data();

        // count nnz
        c_row_ptrs_array.resize_and_reset(m + 1);
        c_row_ptrs = c_row_ptrs_array.get_data();
        cusparse::spgeam_nnz(handle, m, n, a_descr, a_nnz, a_row_ptrs,
                             a_col_idxs, b_descr, b_nnz, b_row_ptrs, b_col_idxs,
                             c_descr, c_row_ptrs, &c_nnz, buffer);

        // accumulate non-zeros
        c_col_idxs_array.resize_and_reset(c_nnz);
        c_vals_array.resize_and_reset(c_nnz);
        c_col_idxs = c_col_idxs_array.get_data();
        c_vals = c_vals_array.get_data();
        cusparse::spgeam(handle, m, n, &valpha, a_descr, a_nnz, a_vals,
                         a_row_ptrs, a_col_idxs, &vbeta, b_descr, b_nnz, b_vals,
                         b_row_ptrs, b_col_idxs, c_descr, c_vals, c_row_ptrs,
                         c_col_idxs, buffer);

        cusparse::destroy(c_descr);
        cusparse::destroy(b_descr);
        cusparse::destroy(a_descr);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
};


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_SPGEAM_KERNEL);


}  // namespace par_ilut_factorization
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
