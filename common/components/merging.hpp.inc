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

namespace detail {


/**
 * @internal
 * The result from the @ref parallel_merge_step function.
 */
template <typename ValueType>
struct merge_result {
    /** The element of a being merged in the current thread. */
    ValueType a_val;
    /** The element of b being merged in the current thread. */
    ValueType b_val;
    /** The index from a that is being merged in the current thread. */
    int a_idx;
    /** The index from b that is being merged in the current thread. */
    int b_idx;
    /** The number of elements from a that have been merged in total. */
    int a_advance;
    /** The number of elements from b that have been merged in total. */
    int b_advance;
};


template <typename ValueType, typename Group>
ValueType shfl_read(Group group, ValueType v, int idx)
{
    // avoid lane ID overflows (especially in the negative direction)
    return group.shfl(v, (idx + group.size()) % group.size());
}


template <typename ValueType>
ValueType checked_load(const ValueType *p, int i, int size)
{
    constexpr auto sentinel = device_numeric_limits<ValueType>::max();
    return i < size ? p[i] : sentinel;
}


}  // namespace detail


/**
 * @internal
 * Warp-parallel merge algorithm that merges the first `warp_size` elements from
 * two ranges, where each warp stores a single element from each range.
 * It assumes that the elements are sorted in ascending order, i.e. for i < j,
 * the value of `a` at thread i is smaller or equal the value at thread j, and
 * the same holds for `b`.
 *
 * @param a  the element from the first range
 * @param b  the element from the second range
 * @param size  the maximum number of elements from both ranges
 * @param group  the cooperative group that executes the merge
 * @return  a structure containing the merge result distributed over the group.
 */
template <typename ValueType, typename Group>
__device__ detail::merge_result<ValueType> group_merge_step(ValueType a,
                                                            ValueType b,
                                                            int size,
                                                            Group group)
{
    // round up to the next power of two
    auto size_pow2 = 1 << (32 - clz(int32(size - 1)));
    // thread i takes care of ith element of the merged sequence
    auto i = group.thread_rank();

    // we want to find the smallest index `x` such that a[x] > b[i - x]
    // this means that `x` gives us the first element of `a` that is no longer
    // part of the output in the range [0...i]. That especially means that
    // a[0...x - 1] and b[0...i - x] form the output range c[0...i]
    // and a[x - 1] and b[i - x] were the last elements to be compared.
    auto max_x = synchronous_binary_search(size_pow2, [&](int x) {
        auto a_remote = shfl_read(group, a, x);
        auto b_remote = shfl_read(group, b, (i - x));

        // `true` sentinel for i - x < 0
        return a_remote > b_remote || x > i;
    });

    auto a_idx = max_x - 1;
    auto b_idx = i - x;
    auto a_val = shfl_read(group, a, a_idx);
    auto b_val = shfl_read(group, b, b_idx);
    auto cmp = a_val < b_val;
    auto a_advance = popcnt(group.ballot(cmp));
    auto b_advance = group.size() - a_advance;

    return {a_val, b_val, a_idx, b_idx, a_advance, b_advance};
}


/**
 * @internal
 * Warp-parallel merge algorithm that merges two sorted ranges of arbitrary
 * size. `merge_fun` will be called for each merged element.
 *
 * @param a  the first range
 * @param a_size the size of the first range
 * @param b  the second range
 * @param b_size the size of the second range
 * @param merge_fn  the callback that is being called for each merged elements.
 *                  It takes three parameters:
 *                  `ValueType val, bool merged_a, IndexType source_index`.
 *                  `val` is the merged value, `merged_a` indicates whether the
 *                  element is from `a` or `b`, and `source_index` is the index
 *                  of `val` in `a` or `b`.
 */
template <typename IndexType, typename ValueType, typename Group,
          typename Callback>
__device__ void group_merge(const ValueType *a, IndexType a_size,
                            const ValueType *b, IndexType b_size, Group group,
                            Callback merge_fn)
{
    auto c_size = a_size + b_size;
    auto a_begin = 0;
    auto b_begin = 0;
    auto c_begin = 0;
    auto lane = group.thread_rank();
    auto a_cur = checked_load(a, lane, a_size);
    auto b_cur = checked_load(b, lane, b_size);
    while (c_begin < c_size) {
        auto merge_size = min(group.size(), c_size - c_begin);
        auto merge_result = group_merge_step(a_cur, b_cur, merge_size, group);
        if (c_begin + lane < c_size) {
            merge_fn(merge_result.a_idx + a_begin, merge_result.a_val,
                     merge_result.b_idx + b_begin, merge_result.b_val,
                     c_begin + lane);
        }
        auto a_advance = merge_result.a_advance;
        auto b_advance = merge_result.b_advance;
        a_begin += a_advance;
        b_begin += b_advance;
        c_begin += group.size();

        // shuffle the unmerged elements to the front
        a_local = warp.shfl_down(a_local, a_advance);
        b_local = warp.shfl_down(b_local, b_advance);
        /*
         * To optimize memory access, we load the new elements for `a` and `b`
         * with a single load instruction:
         * the lower part of the group loads new elements for `a`
         * the upper part of the group loads new elements for `b`
         * `load_lane` is the part-local lane idx
         * The elements for `a` have to be shuffled up afterwards.
         */
        auto load_a = lane < a_advance;
        auto load_lane = load_a ? lane : lane - a_advance;
        auto load_source = load_a ? a : b;
        auto load_begin = load_a ? a_begin + b_advance : b_begin + a_advance;
        auto load_size = load_a ? a_size : b_size;

        auto load_idx = load_begin + load_lane;
        auto loaded = checked_load(load_source, load_idx, load_size);
        // shuffle the `a` values to the end of the warp
        auto lower_loaded = group.shfl_up(loaded, b_advance);
        a_local = lane < b_advance ? a_local : lower_loaded;
        b_local = lane < a_advance ? b_local : loaded;
    }
}