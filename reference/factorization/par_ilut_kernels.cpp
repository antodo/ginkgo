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

#include "core/factorization/par_ilut_kernels.hpp"


#include <algorithm>
#include <unordered_map>
#include <unordered_set>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The parallel ilut factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ilut_factorization {


template <typename ValueType, typename IndexType>
remove_complex<ValueType> threshold_select(
    std::shared_ptr<const ReferenceExecutor> exec, const ValueType *values,
    IndexType size, IndexType rank)
{
    Array<ValueType> data(exec, size);
    std::copy_n(values, size, data.get_data());

    auto begin = data.get_data();
    auto target = begin + rank;
    auto end = begin + size;
    std::nth_element(begin, target, end,
                     [](ValueType a, ValueType b) { return abs(a) < abs(b); });
    return abs(*target);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_THRESHOLD_SELECT_KERNEL);


template <typename ValueType, typename IndexType>
void threshold_filter(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *a,
                      remove_complex<ValueType> threshold,
                      Array<IndexType> &new_row_ptrs_array,
                      Array<IndexType> &new_col_idxs_array,
                      Array<ValueType> &new_vals_array, bool /* is_lower */)
{
    auto num_rows = a->get_size()[0];
    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();
    auto vals = a->get_const_values();

    // first sweep: count nnz for each row
    new_row_ptrs_array.resize_and_reset(num_rows + 1);
    auto new_row_ptrs = new_row_ptrs_array.get_data();
    for (size_type row = 0; row < num_rows; ++row) {
        IndexType count{};
        for (size_type nz = row_ptrs[row]; nz < size_type(row_ptrs[row + 1]);
             ++nz) {
            count += abs(vals[nz]) >= threshold || col_idxs[nz] == row;
        }
        new_row_ptrs[row + 1] = count;
    }

    // build row pointers: exclusive scan (thus the + 1)
    new_row_ptrs[0] = 0;
    std::partial_sum(new_row_ptrs + 1, new_row_ptrs + num_rows + 1,
                     new_row_ptrs + 1);

    // second sweep: accumulate non-zeros
    auto new_nnz = new_row_ptrs[num_rows];
    new_col_idxs_array.resize_and_reset(new_nnz);
    new_vals_array.resize_and_reset(new_nnz);
    auto new_col_idxs = new_col_idxs_array.get_data();
    auto new_vals = new_vals_array.get_data();

    for (size_type row = 0; row < num_rows; ++row) {
        auto new_nz = new_row_ptrs[row];
        for (size_type nz = row_ptrs[row]; nz < size_type(row_ptrs[row + 1]);
             ++nz) {
            if (abs(vals[nz]) >= threshold || col_idxs[nz] == row) {
                new_col_idxs[new_nz] = col_idxs[nz];
                new_vals[new_nz] = vals[nz];
                ++new_nz;
            }
        }
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_THRESHOLD_FILTER_KERNEL);


}  // namespace par_ilut_factorization
}  // namespace reference
}  // namespace kernels
}  // namespace gko
