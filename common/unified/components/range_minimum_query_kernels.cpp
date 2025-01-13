// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/range_minimum_query_kernels.hpp"

#include <limits>

#include <ginkgo/core/base/intrinsics.hpp>

#include "common/unified/base/kernel_launch.hpp"
#include "core/components/bit_packed_storage.hpp"
#include "core/components/range_minimum_query.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace range_minimum_query {


template <typename IndexType>
void compute_lookup_small(std::shared_ptr<const DefaultExecutor> exec,
                          const IndexType* values, IndexType size,
                          block_argmin_storage_type<IndexType>& block_argmin,
                          IndexType* block_min, uint16* block_type)
{
    using tree_index_type = std::decay_t<decltype(*block_type)>;
    using device_lut_type =
        gko::device_block_range_minimum_query_lookup_table<small_block_size>;
    static_assert(device_lut_type::type::num_trees <=
                      std::numeric_limits<tree_index_type>::max(),
                  "block type storage too small");
    constexpr auto collation_width =
        1 << (std::decay_t<decltype(block_argmin)>::bits_per_word_log2 -
              ceil_log2_constexpr(ceil_log2_constexpr(small_block_size)));
    const device_lut_type lut{exec};
    run_kernel(
        exec,
        [] GKO_KERNEL(auto collated_block_idx, auto values, auto block_argmin,
                      auto block_min, auto block_type, auto lut, auto size) {
            constexpr auto infinity = std::numeric_limits<IndexType>::max();
            const auto num_blocks = ceildiv(size, small_block_size);
            for (auto block_idx = collated_block_idx * collation_width;
                 block_idx <
                 std::min<int64>((collated_block_idx + 1) * collation_width,
                                 num_blocks);
                 block_idx++) {
                const auto i = block_idx * small_block_size;
                IndexType local_values[small_block_size];
                int argmin = 0;
#pragma unroll
                for (int local_i = 0; local_i < small_block_size; local_i++) {
                    // use "infinity" as sentinel for minimum computations
                    local_values[local_i] =
                        local_i + i < size ? values[local_i + i] : infinity;
                    if (local_values[local_i] < local_values[argmin]) {
                        argmin = local_i;
                    }
                }
                const auto tree_number = lut->compute_tree_index(local_values);
                const auto min = local_values[argmin];
                // TODO collate these so a single thread handles the argmins for
                // an entire memory word
                block_argmin.set(block_idx, argmin);
                block_min[block_idx] = min;
                block_type[block_idx] =
                    static_cast<tree_index_type>(tree_number);
            }
        },
        ceildiv(ceildiv(size, small_block_size), collation_width), values,
        block_argmin, block_min, block_type, lut.get(), size);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_RANGE_MINIMUM_QUERY_COMPUTE_LOOKUP_SMALL_KERNEL);


template <typename IndexType>
void compute_lookup_large(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType* block_min,
    IndexType num_blocks,
    range_minimum_query_superblocks<IndexType>& superblocks)
{
    using superblock_type = range_minimum_query_superblocks<IndexType>;
    using word_type = typename superblock_type::storage_type;
    // we need to collate all writes that target the same memory word in a
    // single thread
    constexpr auto level0_collation_width = sizeof(word_type) * CHAR_BIT;
    // initialize the first level of blocks
    run_kernel(
        exec,
        [] GKO_KERNEL(auto collated_i, auto block_min, auto superblocks,
                      auto num_blocks) {
            constexpr auto infinity = std::numeric_limits<IndexType>::max();
            for (auto i = collated_i * level0_collation_width;
                 i < std::min<int64>((collated_i + 1) * level0_collation_width,
                                     num_blocks);
                 i++) {
                const auto min1 = block_min[i];
                const auto min2 =
                    i + 1 < num_blocks ? block_min[i + 1] : infinity;
                // we need to use <= here to make sure ties always break to the
                // left
                superblocks.set_block_argmin(0, i, min1 <= min2 ? 0 : 1);
            }
        },
        ceildiv(num_blocks, level0_collation_width), block_min, superblocks,
        num_blocks);
    // we computed argmins for blocks of size 2, now recursively combine them.
    const auto num_levels = superblocks.num_levels();
    for (int block_level = 1; block_level < num_levels; block_level++) {
        const auto block_size =
            superblock_type::block_size_for_level(block_level);
        // we need block_level + 1 bits to represent values of size block_size
        // and round up to the next power of two
        const auto collation_width =
            level0_collation_width / round_up_pow2(block_level + 1);
        run_kernel(
            exec,
            [] GKO_KERNEL(auto collated_i, auto block_level, auto block_min,
                          auto superblocks, auto num_blocks,
                          auto collation_width) {
                const auto block_size =
                    superblock_type::block_size_for_level(block_level);
                for (auto i = collated_i * collation_width;
                     i < std::min<int64>((collated_i + 1) * collation_width,
                                         num_blocks);
                     i++) {
                    const auto i2 = i + block_size / 2;
                    const auto argmin1 =
                        i + superblocks.block_argmin(block_level - 1, i);
                    const auto argmin2 =
                        i2 < num_blocks
                            ? i2 + superblocks.block_argmin(block_level - 1, i2)
                            : argmin1;
                    const auto min1 = block_min[argmin1];
                    const auto min2 = block_min[argmin2];
                    // we need to use <= here to make sure
                    // ties always break to the left
                    superblocks.set_block_argmin(
                        block_level, i,
                        min1 <= min2 ? argmin1 - i : argmin2 - i);
                }
            },
            ceildiv(num_blocks, collation_width), block_level, block_min,
            superblocks, num_blocks, collation_width);
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_RANGE_MINIMUM_QUERY_COMPUTE_LOOKUP_LARGE_KERNEL);


}  // namespace range_minimum_query
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
