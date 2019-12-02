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


#include <algorithm>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/pointer_mode_guard.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/prefix_sum.cuh"
#include "cuda/components/sorting.cuh"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The parallel ilut factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ilut_factorization {


constexpr auto default_block_size = 512;
constexpr auto items_per_thread = 2;


#include "common/factorization/par_ilut_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void ssss_count(const ValueType *values, IndexType size,
                remove_complex<ValueType> *tree, unsigned char *oracles,
                IndexType *partial_counts, IndexType *total_counts)
{
    constexpr auto bucket_count = kernel::searchtree_width;
    auto num_threads_total = ceildiv(size, items_per_thread);
    auto num_blocks = IndexType(ceildiv(num_threads_total, default_block_size));
    // pick sample, build searchtree
    kernel::build_searchtree<<<1, bucket_count>>>(as_cuda_type(values), size,
                                                  tree);
    // determine bucket sizes
    kernel::count_buckets<<<num_blocks, default_block_size>>>(
        as_cuda_type(values), size, tree, partial_counts, oracles,
        items_per_thread);
    // compute prefix sum and total sum over block-local values
    kernel::block_prefix_sum<<<bucket_count, default_block_size>>>(
        partial_counts, total_counts, num_blocks);
    // compute prefix sum over bucket counts
    start_prefix_sum<bucket_count><<<1, bucket_count>>>(
        bucket_count, total_counts, total_counts + bucket_count);
}


template <typename ValueType, typename IndexType>
void ssss_filter(const ValueType *values, IndexType size,
                 const unsigned char *oracles, const IndexType *partial_counts,
                 IndexType bucket, remove_complex<ValueType> *out)
{
    auto num_threads_total = ceildiv(size, items_per_thread);
    auto num_blocks = IndexType(ceildiv(num_threads_total, default_block_size));
    kernel::filter_bucket<<<num_blocks, default_block_size>>>(
        as_cuda_type(values), size, bucket, oracles, partial_counts, out,
        items_per_thread);
}


template <typename ValueType, typename IndexType>
remove_complex<ValueType> threshold_select(
    std::shared_ptr<const CudaExecutor> exec, const ValueType *values,
    IndexType size, IndexType rank)
{
    using AbsType = remove_complex<ValueType>;
    constexpr auto bucket_count = kernel::searchtree_width;
    auto max_num_threads = ceildiv(size, items_per_thread);
    auto max_num_blocks = ceildiv(max_num_threads, default_block_size);

    // we use the last entry to store the total element count
    Array<IndexType> total_counts_array(exec, bucket_count + 1);
    Array<IndexType> partial_counts_array(exec, bucket_count * max_num_blocks);
    Array<unsigned char> oracle_array(exec, size);
    Array<AbsType> tree_array(exec, kernel::searchtree_size);
    auto partial_counts = partial_counts_array.get_data();
    auto total_counts = total_counts_array.get_data();
    auto oracles = oracle_array.get_data();
    auto tree = tree_array.get_data();

    ssss_count(values, size, tree, oracles, partial_counts, total_counts);

    // determine bucket with correct rank
    Array<IndexType> splitter_ranks_array(exec->get_master(),
                                          total_counts_array);
    auto splitter_ranks = splitter_ranks_array.get_const_data();
    auto it = std::upper_bound(splitter_ranks,
                               splitter_ranks + bucket_count + 1, rank);
    auto bucket = IndexType(std::distance(splitter_ranks + 1, it));
    auto bucket_size = splitter_ranks[bucket + 1] - splitter_ranks[bucket];
    rank -= splitter_ranks[bucket];

    Array<AbsType> tmp_out_array(exec, bucket_size);
    Array<AbsType> tmp_in_array(exec, bucket_size);
    auto tmp_out = tmp_out_array.get_data();
    auto tmp_in = tmp_in_array.get_const_data();
    // extract target bucket
    ssss_filter(values, size, oracles, partial_counts, bucket, tmp_out);

    // recursively select from smaller buckets
    int step{};
    while (bucket_size > kernel::basecase_size) {
        std::swap(tmp_out_array, tmp_in_array);
        tmp_out = tmp_out_array.get_data();
        tmp_in = tmp_in_array.get_const_data();

        ssss_count(tmp_in, bucket_size, tree, oracles, partial_counts,
                   total_counts);
        splitter_ranks_array = total_counts_array;
        splitter_ranks = splitter_ranks_array.get_const_data();
        auto it = std::upper_bound(splitter_ranks,
                                   splitter_ranks + bucket_count + 1, rank);
        bucket = IndexType(std::distance(splitter_ranks + 1, it));
        ssss_filter(tmp_in, bucket_size, oracles, partial_counts, bucket,
                    tmp_out);

        rank -= splitter_ranks[bucket];
        bucket_size = splitter_ranks[bucket + 1] - splitter_ranks[bucket];
        // we should never need more than 5 recursion steps, this would mean
        // 256^5 = 2^40. fall back to standard library algorithm in that case.
        ++step;
        if (step > 5) {
            Array<AbsType> cpu_out_array{exec->get_master(), tmp_out_array};
            auto begin = cpu_out_array.get_data();
            auto end = begin + bucket_size;
            auto middle = begin + rank;
            std::nth_element(begin, middle, end);
            return *middle;
        }
    }

    // base case
    Array<AbsType> result_array{exec, 1};
    kernel::basecase_select<<<1, kernel::basecase_block_size>>>(
        tmp_out, bucket_size, rank, result_array.get_data());
    AbsType result{};
    exec->get_master()->copy_from(exec.get(), 1, result_array.get_const_data(),
                                  &result);
    return result;
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_THRESHOLD_SELECT_KERNEL);


template <typename ValueType, typename IndexType>
void threshold_filter(std::shared_ptr<const CudaExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *a,
                      remove_complex<ValueType> threshold,
                      Array<IndexType> &new_row_ptrs_array,
                      Array<IndexType> &new_col_idxs_array,
                      Array<ValueType> &new_vals_array)
{
    auto old_row_ptrs = a->get_const_row_ptrs();
    auto old_col_idxs = a->get_const_col_idxs();
    auto old_vals = a->get_const_values();
    // compute nnz for each row
    auto num_rows = IndexType(a->get_size()[0]);
    auto num_blocks = ceildiv(num_rows, default_block_size);
    new_row_ptrs_array.resize_and_reset(num_rows + 1);
    auto new_row_ptrs = new_row_ptrs_array.get_data();
    kernel::threshold_filter_nnz<<<num_blocks, default_block_size>>>(
        old_row_ptrs, as_cuda_type(old_vals), num_rows, threshold,
        new_row_ptrs);

    // build row pointers
    auto num_row_ptrs = num_rows + 1;
    auto num_reduce_blocks = ceildiv(num_row_ptrs, default_block_size);
    Array<IndexType> block_counts_array(exec, num_blocks);
    auto block_counts = block_counts_array.get_data();

    start_prefix_sum<default_block_size>
        <<<num_reduce_blocks, default_block_size>>>(num_row_ptrs, new_row_ptrs,
                                                    block_counts);
    finalize_prefix_sum<default_block_size>
        <<<num_reduce_blocks, default_block_size>>>(num_row_ptrs, new_row_ptrs,
                                                    block_counts);

    // build matrix
    IndexType num_nnz{};
    exec->get_master()->copy_from(exec.get(), 1, new_row_ptrs + num_rows,
                                  &num_nnz);
    new_col_idxs_array.resize_and_reset(num_nnz);
    new_vals_array.resize_and_reset(num_nnz);
    auto new_col_idxs = new_col_idxs_array.get_data();
    auto new_vals = new_vals_array.get_data();
    kernel::threshold_filter<<<num_blocks, default_block_size>>>(
        old_row_ptrs, old_col_idxs, as_cuda_type(old_vals), num_rows, threshold,
        new_row_ptrs, new_col_idxs, as_cuda_type(new_vals));
}


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
