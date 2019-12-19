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
 * Bitonic sorting operation for two elements.
 *
 * @param reverse  sorts in ascending order if `false` and
 *                 descending order if `true`.
 */
template <typename ValueType>
__device__ void bitonic_cas(ValueType &a, ValueType &b, bool reverse)
{
    auto tmp = a;
    bool cmp = (a < b) != reverse;
    a = cmp ? a : b;
    b = cmp ? b : tmp;
}


/**
 * @internal
 * This is a recursive implementation of a bitonic sorting network,
 * executed sequentially on locally stored data.
 *
 * Based on Batcher, "Sorting Networks and Their Applications", 1968.
 */
template <typename ValueType, int num_elements>
struct bitonic_local {
    using half = bitonic_local<ValueType, num_elements / 2>;
    static_assert(num_elements > 0, "number of elements must be positive");
    static_assert((num_elements & (num_elements - 1)) == 0,
                  "number of elements must be a power of two");

    // merges two bitonic sequences els[0, n / 2), els[n / 2, n)
    __host__ __device__ static void merge(ValueType *els, bool reverse)
    {
        auto els_mid = els + (num_elements / 2);
        for (auto i = 0; i < num_elements / 2; ++i) {
            bitonic_cas(els[i], els_mid[i], reverse);
        }
        half::merge(els, reverse);
        half::merge(els_mid, reverse);
    }

    // sorts an unsorted sequence els [0, n)
    __device__ static void sort(ValueType *els, bool reverse)
    {
        auto els_mid = els + (num_elements / 2);
        // sort first half normally
        half::sort(els, reverse);
        // sort second half reversed
        half::sort(els_mid, !reverse);
        // merge two halves
        merge(els, reverse);
    }
};

template <typename ValueType>
struct bitonic_local<ValueType, 1> {
    // nothing to do for a single element
    __device__ static void merge(ValueType *, bool) {}
    __device__ static void sort(ValueType *, bool) {}
};


/**
 * @internal
 * This is a recursive implementation of a bitonic sorting network,
 * executed in parallel within a warp using lane shuffle instructions.
 *
 * Based on Hou et al., "Fast Segmented Sort on GPUs", 2017.
 */
template <typename ValueType, int num_local, int num_threads>
struct bitonic_warp {
    constexpr static auto num_elements = num_local * num_threads;
    using half = bitonic_warp<ValueType, num_local, num_threads / 2>;
    static_assert(num_threads > 0, "number of threads must be positive");
    static_assert(num_local > 0, "number of local elements must be positive");
    static_assert(
        config::warp_size % num_threads == 0 &&
            num_threads <= config::warp_size,
        "number of threads must be a power of two smaller than warp_size");

    // check if we are in the upper half of all threads in this group
    // this is important as
    // 1. for sorting, we have to reverse the sort order in the upper half
    // 2. for merging, we have to determine for the XOR shuffle if we are
    //    the "smaller" thread, as this thread gets the "smaller" element.
    __device__ static bool upper_half()
    {
        return bool(threadIdx.x & (num_threads / 2));
    }

    __device__ static void merge(ValueType *els, bool reverse)
    {
        auto tile = group::thread_block_tile<num_threads>{};
        auto new_reverse = reverse != upper_half();
        for (auto i = 0; i < num_local; ++i) {
            auto other = tile.shfl_xor(els[i], num_threads / 2);
            bitonic_cas(els[i], other, new_reverse);
        }
        half::merge(els, reverse);
    }

    __device__ static void sort(ValueType *els, bool reverse)
    {
        auto new_reverse = reverse != upper_half();
        half::sort(els, new_reverse);
        merge(els, reverse);
    }
};

template <typename ValueType, int NumLocalElements>
struct bitonic_warp<ValueType, NumLocalElements, 1> {
    using local = bitonic_local<ValueType, NumLocalElements>;
    __device__ static void merge(ValueType *els, bool reverse)
    {
        local::merge(els, reverse);
    }
    __device__ static void sort(ValueType *els, bool reverse)
    {
        local::sort(els, reverse);
    }
};


/**
 * @internal
 * This is a recursive implementation of a bitonic sorting network,
 * executed in parallel in a thread block using shared memory.
 *
 * We use a tiled storage pattern to avoid memory bank collisions on shared
 * memory accesses, see @ref shared_idx.
 */
template <typename ValueType, int num_local, int num_threads, int num_groups,
          int num_total_threads>
struct bitonic_global {
    constexpr static auto num_elements = num_local * num_threads * num_groups;
    using half = bitonic_global<ValueType, num_local, num_threads,
                                num_groups / 2, num_total_threads>;
    static_assert(num_groups > 0, "number of groups must be positive");
    static_assert(num_threads > 0,
                  "number of threads per group must be positive");
    static_assert(num_local > 0, "number of local elements must be positive");
    static_assert(num_total_threads > 0, "number of threads must be positive");
    static_assert(32 % num_groups == 0,
                  "num_groups must be a power of two <= 32");

    __device__ static int shared_idx(int local)
    {
        auto rank = group::this_thread_block().thread_rank();
        // use the same memory-bank to avoid bank conflicts
        return rank + local * num_total_threads;
    }

    // check if we are in the upper half of all groups in this block
    // this is important as for sorting, we have to reverse the sort order in
    // the upper half
    __device__ static bool upper_half()
    {
        auto rank = group::this_thread_block().thread_rank();
        return bool(rank & (num_groups * num_threads / 2));
    }

    __device__ static void merge(ValueType *local_els, ValueType *shared_els,
                                 bool reverse)
    {
        group::this_thread_block().sync();
        auto upper_shared_els = shared_els + (num_groups * num_threads / 2);
        // only the lower group executes the CAS
        if (!upper_half()) {
            for (auto i = 0; i < num_local; ++i) {
                auto j = shared_idx(i);
                bitonic_cas(shared_els[j], upper_shared_els[j], reverse);
            }
        }
        half::merge(local_els, shared_els, reverse);
    }

    __device__ static void sort(ValueType *local_els, ValueType *shared_els,
                                bool reverse)
    {
        auto new_reverse = reverse != upper_half();
        half::sort(local_els, shared_els, new_reverse);
        merge(local_els, shared_els, reverse);
    }
};

template <typename ValueType, int num_local, int num_threads,
          int num_total_threads>
struct bitonic_global<ValueType, num_local, num_threads, 1, num_total_threads> {
    using warp = bitonic_warp<ValueType, num_local, num_threads>;

    __device__ static int shared_idx(int local)
    {
        // use the indexing from the general struct
        return bitonic_global<ValueType, num_local, num_threads, 2,
                              num_total_threads>::shared_idx(local);
    }

    __device__ static void merge(ValueType *local_els, ValueType *shared_els,
                                 bool reverse)
    {
        group::this_thread_block().sync();
        for (auto i = 0; i < num_local; ++i) {
            local_els[i] = shared_els[shared_idx(i)];
        }
        warp::merge(local_els, reverse);
        for (auto i = 0; i < num_local; ++i) {
            shared_els[shared_idx(i)] = local_els[i];
        }
    }

    __device__ static void sort(ValueType *local_els, ValueType *shared_els,
                                bool reverse)
    {
        auto rank = group::this_thread_block().thread_rank();
        // This is the first step, so we don't need to load from shared memory
        warp::sort(local_els, reverse);
        // store the sorted elements in shared memory
        for (auto i = 0; i < num_local; ++i) {
            shared_els[shared_idx(i)] = local_els[i];
        }
    }
};


}  // namespace detail


/**
 * @internal
 *
 * This function sorts elements within a thread block.
 *
 * It takes a local array of elements and the pointer to a shared buffer of size
 * `num_elements` as input. After the execution, the thread with rank `i` in the
 * thread block (determined by `group::this_thread_block().thread_rank()`) has
 * the elements at index `num_local * i` up to `num_local * i + (num_local - 1)`
 * in the sorted sequence stored in its `local_elements` at index 0 up to
 * `num_local - 1`.
 *
 * @note The shared-memory buffer uses a striped layout to limit bank
 *       collisions, so it should not directly be used to access elements from
 *       the sorted sequence. If `num_elements <= num_local * warp_size`, the
 *       algorithm doesn't use/need the shared-memory buffer, so it can be null.
 *
 * @param local_elements  the `num_local` input/output elements from this
 *                        thread.
 * @param shared_elements  the shared-memory buffer of size `num_elements`
 * @tparam num_elements  the number of elements - it must be a power of two!
 * @tparam num_local  the number of elements stored per thread - it must be a
 *                    power of two!
 * @tparam ValueType  the type of the elements to be sorted - it must implement
 *                    the less-than operator!
 */
template <int num_elements, int num_local, typename ValueType>
__device__ void bitonic_sort(ValueType *local_elements,
                             ValueType *shared_elements)
{
    constexpr auto num_threads = num_elements / num_local;
    constexpr auto num_warps = num_threads / config::warp_size;
    static_assert(num_threads <= config::max_block_size,
                  "bitonic_sort exceeds thread block");
    if (num_warps > 1) {
        // these checks are necessary since the `if` is not evaluated at
        // compile-time so even though the branch is never taken, it still gets
        // instantiated and must thus compile.
        constexpr auto _num_warps = num_warps <= 1 ? 1 : num_warps;
        constexpr auto _num_threads =
            num_threads <= config::warp_size ? config::warp_size : num_threads;
        detail::bitonic_global<ValueType, num_local, config::warp_size,
                               _num_warps, _num_threads>::sort(local_elements,
                                                               shared_elements,
                                                               false);
    } else {
        constexpr auto _num_threads = num_warps > 1 ? 1 : num_threads;
        detail::bitonic_warp<ValueType, num_local, _num_threads>::sort(
            local_elements, false);
    }
}


/**
 * @internal
 * Generic binary search that finds the first index where a predicate is true.
 * It assumes that the predicate partitions the range [offset, offset + length)
 * into two subranges [offset, middle), [middle, offset + length) such that
 * the predicate is `false` for all elements in the first range and `true` for
 * all elements in the second range. `middle` is called the partition point.
 * The implementation is based on Stepanov & McJones, "Elements of Programming".
 *
 * @param offset  the starting index of the partitioned range
 * @param length  the length of the partitioned range
 * @param p  the predicate to be evaluated on the range - it should not have
 *           side-effects and map from `IndexType` to `bool`
 * @returns  the index of `middle`, i.e., the partition point
 */
template <typename IndexType, typename Predicate>
__device__ IndexType binary_search(IndexType offset, IndexType length,
                                   Predicate p)
{
    while (length > 0) {
        auto half_length = length / 2;
        auto mid = offset + half_length;
        auto pred = p(mid);
        length = pred ? half_length : length - (half_length + 1);
        offset = pred ? offset : mid + 1;
    }
    return offset;
}


/**
 * @internal
 * Generic implementation of a fixed-size binary search.
 * The implementation makes sure that the number of predicate evaluations only
 * depends on `length` and not on the actual position of the partition point.
 *
 * @tparam size  the length of the partitioned range - must be a power of two
 * @param p  the predicate to be evaluated on the range - it should not have
 *           side-effects and map from `int` to `bool`
 * @returns  the index of `middle`, i.e., the partition point
 */
template <int size, typename Predicate>
__device__ int synchronous_binary_search(Predicate p)
{
    auto begin = 0;
    static_assert(size > 0, "size must be positive");
    static_assert(!(size & (size - 1)), "size must be a power of two");
#pragma unroll
    for (auto cur_size = size; cur_size > 1; cur_size /= 2) {
        auto half_size = cur_size / 2;
        auto mid = begin + half_size;
        // invariant: [begin, begin + cur_size] contains partition point
        begin = p(mid) ? begin : mid;
    }
    // cur_size is now 1, so the partition point is either begin or begin + 1
    return p(begin) ? begin : begin + 1;
}


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