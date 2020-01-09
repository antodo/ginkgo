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
 * The result from the @ref warp_merge_step function.
 */
template <typename ValueType>
struct warp_merge_result {
    /** true iff the element at this thread originates from sequence `a`. */
    bool merged_a;
    /** the warp lane index from which the element at this thread originates. */
    int source_index;
    /** the value of the element at this thread. */
    ValueType value;
    /** how many elements of `a` did we merge? */
    int a_advance;
};


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
 */
template <typename ValueType>
__device__ detail::warp_merge_result<ValueType> warp_merge_step(ValueType a,
                                                                ValueType b)
{
    auto warp = group::thread_block_tile<config::warp_size>();
    // thread i takes care of the diagonal (0, i) -> (i, 0)
    auto diag = threadIdx.x % config::warp_size;
    auto a_pos = [&](int i) {
        // avoid out-of-bounds lane accesses
        return (config::warp_size + diag - i) % config::warp_size;
    };
    auto b_pos = [&](int i) { return i; };

    // find the intersection of the diagonal with the merge path
    // we need to "extend" the diagonals such that they all have the same size
    // otherwise not all threads would participate in the shuffle.
    auto intersection =
        synchronous_binary_search<config::warp_size>([&](int d) {
            auto a_remote = warp.shfl(a, a_pos(d));
            auto b_remote = warp.shfl(b, b_pos(d));
            // outside the diagonal, the predicate must be true (sentinel)
            return a_remote < b_remote || d > diag;
        });
    // determine if we merged a or b:
    auto intersection_prev = warp.shfl_up(intersection, 1);
    intersection_prev = diag == 0 ? 0 : intersection_prev;
    auto a_pos_int = a_pos(intersection_prev);
    auto b_pos_int = b_pos(intersection_prev);
    // the intersection index is equal to b_pos
    auto merged_a = intersection_prev == intersection;

    // fetch the corresponding values of a and b
    auto a_remote = warp.shfl(a, a_pos_int);
    auto b_remote = warp.shfl(b, b_pos_int);
    // assert a_remove < b_remote

    // merge them at the current position
    detail::warp_merge_result<ValueType> result{};
    result.merged_a = merged_a;
    result.source_index = merged_a ? a_pos_int : b_pos_int;
    result.value = merged_a ? a_remote : b_remote;
    result.a_advance =
        __popc(warp.ballot(merged_a));  // TODO replace by shuffle
    return result;
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
template <typename IndexType, typename ValueType, typename Callback>
__device__ void warp_merge(const ValueType *a, IndexType a_size,
                           const ValueType *b, IndexType b_size,
                           Callback merge_fn)
{
    constexpr auto sentinel = device_numeric_limits<ValueType>::max();
    auto warp = group::thread_block_tile<config::warp_size>();
    auto c_size = a_size + b_size;
    auto a_begin = 0;
    auto b_begin = 0;
    auto c_begin = 0;
    auto lane = threadIdx.x % config::warp_size;
    auto checked_load = [](const int *p, int i, int size) {
        return i < size ? p[i] : sentinel;
    };
    auto a_local = checked_load(a, lane, a_size);
    auto b_local = checked_load(b, lane, b_size);
    while (c_begin < c_size) {
        auto merge_result = warp_merge_step(a_local, b_local);
        if (c_begin + lane < c_size) {
            auto source_idx = merge_result.source_index + merge_result.merged_a
                                  ? a_begin
                                  : b_begin;
            merge_fn(merge_result.value, merge_result.merged_a, source_idx);
        }
        auto a_advance = merge_result.a_advance;
        auto b_advance = config::warp_size - a_advance;
        a_begin += a_advance;
        b_begin += b_advance;
        c_begin += a_advance + b_advance;

        // shuffle the unmerged elements to the front
        a_local = warp.shfl_down(a_local, a_advance);
        b_local = warp.shfl_down(b_local, b_advance);
        /*
         * To optimize memory access, we load the new elements for `a` and `b`
         * with a single load instruction:
         * the lower part of the warp loads new elements for `a`
         * the upper part of the warp loads new elements for `b`
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
        auto lower_loaded = warp.shfl_up(loaded, b_advance);
        a_local = lane < b_advance ? a_local : lower_loaded;
        b_local = lane < a_advance ? b_local : loaded;
    }
}