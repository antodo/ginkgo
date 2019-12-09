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

namespace kernel {


constexpr auto searchtree_height = 8;
constexpr auto searchtree_width = 1 << searchtree_height;
constexpr auto searchtree_inner_size = searchtree_width - 1;
constexpr auto searchtree_size = searchtree_width + searchtree_inner_size;

constexpr auto oversampling_factor = 4;
constexpr auto sample_size = searchtree_width * oversampling_factor;

constexpr auto basecase_size = 1024;
constexpr auto basecase_local_size = 4;
constexpr auto basecase_block_size = basecase_size / basecase_local_size;


// must be launched with one thread block and block size == searchtree_width
template <typename IndexType, typename ValueType>
__global__ __launch_bounds__(searchtree_width) void build_searchtree(
    const ValueType *input, IndexType size,
    remove_complex<ValueType> *tree_output)
{
    using AbsType = remove_complex<ValueType>;
    auto idx = threadIdx.x;
    AbsType samples[oversampling_factor];
    auto stride = size / sample_size;
#pragma unroll
    for (auto i = 0; i < oversampling_factor; ++i) {
        auto lidx = idx * oversampling_factor + i;
        auto val = input[IndexType(lidx * stride)];
        samples[i] = abs(val);
    }
    __shared__ AbsType sh_samples[sample_size];
    bitonic_sort<sample_size, oversampling_factor>(samples, sh_samples);
    if (threadIdx.x > 0) {
        // root has level 0
        auto level = searchtree_height - __ffs(threadIdx.x);
        // we get the in-level index by removing trailing 10000...
        auto idx_in_level = threadIdx.x >> __ffs(threadIdx.x);
        // we get the global index by adding previous levels
        auto previous_levels = (1 << level) - 1;
        tree_output[idx_in_level + previous_levels] = samples[0];
    }
    tree_output[threadIdx.x + searchtree_inner_size] = samples[0];
}


// must be launched with default_block_size >= searchtree_width
template <typename IndexType, typename ValueType>
__global__ __launch_bounds__(default_block_size) void count_buckets(
    const ValueType *input, IndexType size,
    const remove_complex<ValueType> *tree, IndexType *counter,
    unsigned char *oracles, int items_per_thread)
{
    // load tree into shared memory, initialize counters
    __shared__ remove_complex<ValueType> sh_tree[searchtree_inner_size];
    __shared__ IndexType sh_counter[searchtree_width];
    if (threadIdx.x < searchtree_inner_size) {
        sh_tree[threadIdx.x] = tree[threadIdx.x];
    }
    if (threadIdx.x < searchtree_width) {
        sh_counter[threadIdx.x] = 0;
    }
    group::this_thread_block().sync();

    // work distribution: each thread block gets a consecutive index range
    auto begin =
        threadIdx.x + default_block_size * blockIdx.x * items_per_thread;
    IndexType block_end =
        default_block_size * (blockIdx.x + 1) * items_per_thread;
    auto end = min(block_end, size);
    for (IndexType i = begin; i < end; i += default_block_size) {
        // traverse the search tree with the input element
        auto el = abs(input[i]);
        IndexType tree_idx{};
#pragma unroll
        for (auto level = 0; level < searchtree_height; ++level) {
            auto cmp = !(el < sh_tree[tree_idx]);
            tree_idx = 2 * tree_idx + 1 + cmp;
        }
        // increment the bucket counter and store the bucket index
        uint32 bucket = tree_idx - searchtree_inner_size;
        atomic_add<IndexType>(sh_counter + bucket, 1);
        oracles[i] = bucket;
    }
    group::this_thread_block().sync();

    // write back the block-wide counts to global memory
    if (threadIdx.x < searchtree_width) {
        counter[blockIdx.x + threadIdx.x * gridDim.x] = sh_counter[threadIdx.x];
    }
}


// must be launched with default_block_size threads per block
template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void block_prefix_sum(
    IndexType *counters, IndexType *totals, IndexType num_blocks)
{
    constexpr auto num_warps = default_block_size / config::warp_size;
    static_assert(num_warps < config::warp_size,
                  "block size needs to be smaller");
    __shared__ IndexType warp_sums[num_warps];

    auto block = group::this_thread_block();

    auto bucket = blockIdx.x;
    auto local_counters = counters + num_blocks * bucket;
    auto work_per_warp = ceildiv(num_blocks, config::warp_size);
    auto warp_idx = threadIdx.x / config::warp_size;
    auto warp_lane = threadIdx.x % config::warp_size;

    // compute prefix sum over warp-sized blocks
    IndexType total{};
    auto base_idx = warp_idx * work_per_warp * config::warp_size;
    for (auto step = 0; step < work_per_warp; ++step) {
        auto idx = warp_lane + step * config::warp_size + base_idx;
        auto val = idx < num_blocks ? local_counters[idx] : zero<IndexType>();
        IndexType warp_total{};
        IndexType warp_prefix{};
        // compute inclusive prefix sum
        warp_prefix_sum<false>(val, warp_prefix, warp_total);

        if (idx < num_blocks) {
            local_counters[idx] = warp_prefix + total;
        }
        total += warp_total;
    }

    // store total sum
    if (warp_lane == 0) {
        warp_sums[warp_idx] = total;
    }

    // compute prefix sum over all warps in a single warp
    block.sync();
    if (warp_idx == 0) {
        auto in_bounds = warp_lane < num_warps;
        auto val = in_bounds ? warp_sums[warp_lane] : zero<IndexType>();
        IndexType prefix_sum{};
        IndexType total_sum{};
        // compute inclusive prefix sum
        warp_prefix_sum<false>(val, prefix_sum, total_sum);
        if (in_bounds) {
            warp_sums[warp_lane] = prefix_sum;
        }
        if (warp_lane == 0) {
            totals[bucket] = total_sum;
        }
    }

    // add block prefix sum to each warp's block of data
    block.sync();
    auto warp_prefixsum = warp_sums[warp_idx];
    for (auto step = 0; step < work_per_warp; ++step) {
        auto idx = warp_lane + step * config::warp_size + base_idx;
        auto val = idx < num_blocks ? local_counters[idx] : zero<IndexType>();
        if (idx < num_blocks) {
            local_counters[idx] += warp_prefixsum;
        }
    }
}


// must be launched with default_block_size >= searchtree_width
template <typename IndexType, typename ValueType>
__global__ __launch_bounds__(default_block_size) void filter_bucket(
    const ValueType *input, IndexType size, unsigned char bucket,
    const unsigned char *oracles, const IndexType *block_offsets,
    remove_complex<ValueType> *output, int items_per_thread)
{
    // initialize the counter with the block prefix sum.
    __shared__ IndexType counter;
    if (threadIdx.x == 0) {
        counter = block_offsets[blockIdx.x + bucket * gridDim.x];
    }
    group::this_thread_block().sync();

    // same work-distribution as in count_buckets
    auto begin =
        threadIdx.x + default_block_size * blockIdx.x * items_per_thread;
    IndexType block_end =
        default_block_size * (blockIdx.x + 1) * items_per_thread;
    auto end = min(block_end, size);
    for (IndexType i = begin; i < end; i += default_block_size) {
        // only copy the element when it belongs to the target bucket
        auto found = bucket == oracles[i];
        auto ofs = atomic_add<IndexType>(&counter, found);
        if (found) {
            output[ofs] = abs(input[i]);
        }
    }
}


template <typename IndexType, typename ValueType>
__global__ __launch_bounds__(basecase_block_size) void basecase_select(
    const ValueType *input, IndexType size, IndexType rank, ValueType *out)
{
    constexpr auto sentinel = device_numeric_limits<ValueType>::inf;
    ValueType local[basecase_local_size];
    __shared__ ValueType sh_local[basecase_size];
    for (int i = 0; i < basecase_local_size; ++i) {
        auto idx = threadIdx.x + i * basecase_block_size;
        local[i] = idx < size ? input[idx] : sentinel;
    }
    bitonic_sort<basecase_size, basecase_local_size>(local, sh_local);
    if (threadIdx.x == rank / basecase_local_size) {
        *out = local[rank % basecase_local_size];
    }
}


template <typename IndexType, typename ValueType>
__global__ __launch_bounds__(default_block_size) void threshold_filter_nnz(
    const IndexType *row_ptrs, const ValueType *vals, IndexType num_rows,
    remove_complex<ValueType> threshold, IndexType *nnz, bool is_lower)
{
    auto row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= num_rows) {
        return;
    }

    IndexType count{};
    // ignoring diagonal entries:
    // lower triangular part has the diagonal last
    // upper triangular part has the diagonal first
    auto begin = row_ptrs[row] + !is_lower;
    auto end = row_ptrs[row + 1] - is_lower;
    for (auto i = begin; i < end; ++i) {
        count += abs(vals[i]) >= threshold;
    }
    // add the diagonal entry
    nnz[row] = count + 1;
}


template <typename IndexType, typename ValueType>
__global__ __launch_bounds__(default_block_size) void threshold_filter(
    const IndexType *old_row_ptrs, const IndexType *old_col_idxs,
    const ValueType *old_vals, IndexType num_rows,
    remove_complex<ValueType> threshold, const IndexType *new_row_ptrs,
    IndexType *new_col_idxs, ValueType *new_vals, bool is_lower)
{
    auto row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= num_rows) {
        return;
    }

    IndexType count{};
    // ignoring diagonal entries:
    // lower triangular part has the diagonal last
    // upper triangular part has the diagonal first
    auto new_begin = new_row_ptrs[row] + !is_lower;
    auto new_end = new_row_ptrs[row + 1] - is_lower;
    auto begin = new_row_ptrs[row] + !is_lower;
    auto end = new_row_ptrs[row + 1] - is_lower;
    for (auto i = begin; i < end; ++i) {
        auto pred = abs(old_vals[i]) >= threshold;
        if (pred) {
            new_col_idxs[count + new_begin] = old_col_idxs[i];
            new_vals[count + new_begin] = old_vals[i];
        }
        count += pred;
    }
    // add diagonal
    auto in_diag = is_lower ? end : begin - 1;
    auto out_diag = is_lower ? new_end : new_begin - 1;
    new_col_idxs[out_diag] = old_col_idxs[in_diag];
    new_vals[out_diag] = old_vals[in_diag];
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void spgeam_nnz(
    const IndexType *a_row_ptrs, const IndexType *a_col_idxs,
    const IndexType *b_row_ptrs, const IndexType *b_col_idxs,
    IndexType num_rows, IndexType *nnz)
{
    constexpr auto sentinel = device_numeric_limits<IndexType>::max;
    auto row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= num_rows) {
        return;
    }

    auto a_begin = a_row_ptrs[row];
    auto b_begin = b_row_ptrs[row];
    auto a_end = a_row_ptrs[row + 1];
    auto b_end = b_row_ptrs[row + 1];
    auto a_size = a_end - a_begin;
    auto b_size = b_end - b_begin;
    auto a_ptr = a_begin;
    auto b_ptr = b_begin;
    auto a_col = a_ptr < a_end ? a_col_idxs[a_ptr] : sentinel;
    auto b_col = b_ptr < b_end ? b_col_idxs[b_ptr] : sentinel;
    IndexType count{};
    IndexType i{};
    while (i < a_size + b_size) {
        auto advance_a = (a_col <= b_col);
        auto advance_b = (b_col <= a_col);
        i += advance_a + advance_b;
        a_ptr += advance_a;
        b_ptr += advance_b;
        if (advance_a) {
            a_col = a_ptr < a_end ? a_col_idxs[a_ptr] : sentinel;
        }
        if (advance_b) {
            b_col = b_ptr < b_end ? b_col_idxs[b_ptr] : sentinel;
        }
        ++count;
    }
    nnz[row] = count;
}


template <typename IndexType, typename ValueType>
__global__ __launch_bounds__(default_block_size) void spgeam(
    ValueType alpha, const IndexType *a_row_ptrs, const IndexType *a_col_idxs,
    const ValueType *a_vals, ValueType beta, const IndexType *b_row_ptrs,
    const IndexType *b_col_idxs, const ValueType *b_vals, IndexType num_rows,
    const IndexType *c_row_ptrs, IndexType *c_col_idxs, ValueType *c_vals)
{
    constexpr auto sentinel = device_numeric_limits<IndexType>::max;
    auto row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= num_rows) {
        return;
    }

    auto a_begin = a_row_ptrs[row];
    auto b_begin = b_row_ptrs[row];
    auto c_begin = c_row_ptrs[row];
    auto a_end = a_row_ptrs[row + 1];
    auto b_end = b_row_ptrs[row + 1];
    auto a_size = a_end - a_begin;
    auto b_size = b_end - b_begin;
    auto a_ptr = a_begin;
    auto b_ptr = b_begin;
    auto a_col = a_ptr < a_end ? a_col_idxs[a_ptr] : sentinel;
    auto b_col = b_ptr < b_end ? b_col_idxs[b_ptr] : sentinel;
    IndexType count{};
    IndexType i{};
    IndexType c_col{};
    while (i < a_size + b_size) {
        auto advance_a = (a_col <= b_col);
        auto advance_b = (b_col <= a_col);
        c_col = advance_a ? a_col : b_col;
        i += advance_a + advance_b;
        a_ptr += advance_a;
        b_ptr += advance_b;
        ValueType c_val{};
        if (advance_a) {
            a_col = a_ptr < a_end ? a_col_idxs[a_ptr] : sentinel;
            c_val += alpha * a_vals[a_ptr - 1];
        }
        if (advance_b) {
            b_col = b_ptr < b_end ? b_col_idxs[b_ptr] : sentinel;
            c_val += beta * b_vals[b_ptr - 1];
        }
        c_col_idxs[c_begin + count] = c_col;
        c_vals[c_begin + count] = c_val;
        ++count;
    }
}


}  // namespace kernel