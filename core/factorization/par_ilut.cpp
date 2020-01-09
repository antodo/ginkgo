/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/core/factorization/par_ilut.hpp>


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/factorization/par_ilu_kernels.hpp"
#include "core/factorization/par_ilut_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"


namespace gko {
namespace factorization {
namespace par_ilut_factorization {


GKO_REGISTER_OPERATION(threshold_filter,
                       par_ilut_factorization::threshold_filter);
GKO_REGISTER_OPERATION(spgeam, par_ilut_factorization::spgeam);


GKO_REGISTER_OPERATION(initialize_row_ptrs_l_u,
                       par_ilu_factorization::initialize_row_ptrs_l_u);
GKO_REGISTER_OPERATION(initialize_l_u, par_ilu_factorization::initialize_l_u);
GKO_REGISTER_OPERATION(compute_l_u_factors,
                       par_ilu_factorization::compute_l_u_factors);


GKO_REGISTER_OPERATION(csr_transpose, csr::transpose);
GKO_REGISTER_OPERATION(spgemm, csr::spgemm);


}  // namespace par_ilut_factorization

template <typename ValueType, typename IndexType>
class ParIlutState {
    // system matrix A in COO format with additional zeros
    Array<ValueType> a_vals;
    Array<IndexType> a_row_idxs;
    Array<IndexType> a_col_idxs;
    // current lower factor L
    Array<IndexType> l_row_ptrs;
    Array<IndexType> l_col_idxs;
    Array<ValueType> l_vals;
    // current upper factor U
    Array<IndexType> u_row_ptrs;
    Array<IndexType> u_col_idxs;
    Array<ValueType> u_vals;
    // current product LU
    Array<IndexType> lu_row_ptrs;
    Array<IndexType> lu_col_idxs;
    Array<ValueType> lu_vals;
    // temporary lower factor L' before filtering
    Array<IndexType> lt_row_ptrs;
    Array<IndexType> lt_col_idxs;
    Array<ValueType> lt_vals;
    // temporary upper factor U' before filtering
    Array<IndexType> ut_row_ptrs;
    Array<IndexType> ut_col_idxs;
    Array<ValueType> ut_vals;

public:
    std::unique_ptr<Composition<ValueType>> to_factors() {}
};

void iterate()

    template <typename ValueType, typename IndexType>
    std::unique_ptr<Composition<ValueType>> ParIlut<ValueType, IndexType>::
        generate_l_u(const std::shared_ptr<const LinOp> &system_matrix) const
{
    using CsrMatrix = matrix::Csr<ValueType, IndexType>;
    using CooMatrix = matrix::Coo<ValueType, IndexType>;

    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

    const auto exec = this->get_executor();
    const auto host_exec = exec->get_master();

    // Only copies the matrix if it is not on the same executor or was not in
    // the right format. Throws an exception if it is not convertable.
    std::unique_ptr<CsrMatrix> csr_system_matrix_unique_ptr{};
    auto csr_system_matrix =
        dynamic_cast<const CsrMatrix *>(system_matrix.get());
    if (csr_system_matrix == nullptr ||
        csr_system_matrix->get_executor() != exec) {
        csr_system_matrix_unique_ptr = CsrMatrix::create(exec);
        as<ConvertibleTo<CsrMatrix>>(system_matrix.get())
            ->convert_to(csr_system_matrix_unique_ptr.get());
        csr_system_matrix = csr_system_matrix_unique_ptr.get();
    }
    // If it needs to be sorted, copy it if necessary and sort it
    if (!_parameters.skip_sorting) {
        if (csr_system_matrix_unique_ptr == nullptr) {
            csr_system_matrix_unique_ptr = CsrMatrix::create(exec);
            csr_system_matrix_unique_ptr->copy_from(csr_system_matrix);
        }
        csr_system_matrix_unique_ptr->sort_by_column_index();
        csr_system_matrix = csr_system_matrix_unique_ptr.get();
    }

    const auto matrix_size = csr_system_matrix->get_size();
    const auto number_rows = matrix_size[0];
    Array<IndexType> l_row_ptrs{exec, number_rows + 1};
    Array<IndexType> u_row_ptrs{exec, number_rows + 1};
    exec->run(par_ilu_factorization::make_initialize_row_ptrs_l_u(
        csr_system_matrix, l_row_ptrs.get_data(), u_row_ptrs.get_data()));

    IndexType l_nnz_it;
    IndexType u_nnz_it;
    // Since nnz is always at row_ptrs[m], it can be extracted easily
    host_exec->copy_from(exec.get(), 1, l_row_ptrs.get_data() + number_rows,
                         &l_nnz_it);
    host_exec->copy_from(exec.get(), 1, u_row_ptrs.get_data() + number_rows,
                         &u_nnz_it);
    auto l_nnz = static_cast<size_type>(l_nnz_it);
    auto u_nnz = static_cast<size_type>(u_nnz_it);

    // Since `row_ptrs` of L and U is already created, the matrix can be
    // directly created with it
    Array<IndexType> l_col_idxs_array{exec, l_nnz};
    Array<ValueType> l_vals_array{exec, l_nnz};
    Array<IndexType> u_col_idxs_array{exec, u_nnz};
    Array<ValueType> u_vals_array{exec, u_nnz};

    exec->run(par_ilu_factorization::make_initialize_l_u(
        csr_system_matrix, l_factor.get(), u_factor.get()));

    for (size_type iteration = 0; iteration < _parameters.iterations;
         ++iteration) {
        exec->run(par_ilut_factorization::make_)
    }
}

}  // namespace factorization
}  // namespace gko
