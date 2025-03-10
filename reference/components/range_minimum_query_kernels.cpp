// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/range_minimum_query_kernels.hpp"

#include <limits>

#include "core/base/intrinsics.hpp"
#include "core/components/bit_packed_storage.hpp"
#include "core/components/range_minimum_query.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace range_minimum_query {


template <typename IndexType>
void compute_lookup_inside_blocks(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType* values,
    IndexType size, bit_packed_span<int, IndexType, uint32>& block_argmin,
    IndexType* block_min, uint16* block_tree_index)
{
    using rmq_type = gko::range_minimum_query<IndexType>;
    constexpr auto block_size = rmq_type::block_size;
    using tree_index_type = std::decay_t<decltype(*block_tree_index)>;
    using lut_type = typename rmq_type::block_lut_view_type;
    lut_type table;
    static_assert(
        lut_type::num_trees <= std::numeric_limits<tree_index_type>::max(),
        "block type storage too small");
    for (IndexType i = 0; i < size; i += block_size) {
        IndexType local_values[block_size];
        for (int local_i = 0; local_i < block_size; local_i++) {
            // use "infinity" as sentinel for minimum computations
            local_values[local_i] = local_i + i < size
                                        ? values[local_i + i]
                                        : std::numeric_limits<IndexType>::max();
        }
        const auto tree_number = table.compute_tree_index(local_values);
        const auto min_it =
            std::min_element(local_values, local_values + block_size);
        const auto min_idx =
            static_cast<uint32>(std::distance(local_values, min_it));
        const auto block_idx = i / block_size;
        block_argmin.set(block_idx, min_idx);
        block_min[block_idx] = *min_it;
        block_tree_index[block_idx] = static_cast<tree_index_type>(tree_number);
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_RANGE_MINIMUM_QUERY_COMPUTE_LOOKUP_SMALL_KERNEL);


template <typename IndexType>
void compute_lookup_across_blocks(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType* block_min,
    IndexType num_blocks,
    device_range_minimum_query_superblocks<IndexType>& superblocks)
{
    using superblock_type = device_range_minimum_query_superblocks<IndexType>;
    constexpr auto infinity = std::numeric_limits<IndexType>::max();
    if (num_blocks < 2) {
        return;
    }
    // initialize the first level of blocks
    for (const auto i : irange{num_blocks}) {
        const auto min1 = block_min[i];
        const auto min2 = i + 1 < num_blocks ? block_min[i + 1] : infinity;
        // we need to use <= here to make sure ties always break to the left
        superblocks.set_block_argmin(0, i, min1 <= min2 ? 0 : 1);
    }
    // we computed argmins for blocks of size 2, now recursively combine them.
    const auto num_levels = superblocks.num_levels();
    for (const auto block_level : irange{1, num_levels}) {
        const auto block_size =
            superblock_type::block_size_for_level(block_level);
        for (const auto i : irange{num_blocks}) {
            const auto i2 = i + block_size / 2;
            const auto argmin1 =
                i + superblocks.block_argmin(block_level - 1, i);
            const auto argmin2 =
                i2 < num_blocks
                    ? i2 + superblocks.block_argmin(block_level - 1, i2)
                    : argmin1;
            const auto min1 = block_min[argmin1];
            const auto min2 = block_min[argmin2];
            // we need to use <= here to make sure ties always break to the left
            superblocks.set_block_argmin(
                block_level, i, min1 <= min2 ? argmin1 - i : argmin2 - i);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_RANGE_MINIMUM_QUERY_COMPUTE_LOOKUP_LARGE_KERNEL);


}  // namespace range_minimum_query
}  // namespace reference
}  // namespace kernels
}  // namespace gko
