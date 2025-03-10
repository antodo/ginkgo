// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/index_map_kernels.hpp"

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <thrust/unique.h>

#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/cuda_hip/base/thrust.hpp"
#include "common/cuda_hip/components/atomic.hpp"
#include "common/cuda_hip/components/searching.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace index_map {


/**
 * This struct is necessary, since the `transform_output_iterator` seemingly
 * doesn't support non-copyable tranfsorm function (this excludes lambdas)
 */
template <typename LocalIndexType, typename GlobalIndexType>
struct transform_output {
    transform_output(const GlobalIndexType* range_bounds_,
                     const LocalIndexType* range_starting_indices_)
        : range_bounds(range_bounds_),
          range_starting_indices(range_starting_indices_)
    {}

    template <typename T>
    __host__ __device__ thrust::tuple<GlobalIndexType, LocalIndexType>
    operator()(const T& t)
    {
        auto gid = thrust::get<0>(t);
        auto rid = thrust::get<1>(t);
        return thrust::make_tuple(gid, map_to_local(gid, rid));
    }

    __host__ __device__ LocalIndexType map_to_local(const GlobalIndexType index,
                                                    const size_type range_id)
    {
        return static_cast<LocalIndexType>(index - range_bounds[range_id]) +
               range_starting_indices[range_id];
    };

    const GlobalIndexType* range_bounds;
    const LocalIndexType* range_starting_indices;
};


template <typename LocalIndexType, typename GlobalIndexType>
array<size_type> compute_range_ids(
    std::shared_ptr<const DefaultExecutor> exec,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        part,
    const array<GlobalIndexType>& idxs)
{
    const auto* range_bounds = part->get_range_bounds();
    const auto num_ranges = part->get_num_ranges();
    auto input_size = idxs.get_size();
    auto idxs_ptr = idxs.get_const_data();

    auto policy = thrust_policy(exec);

    array<size_type> range_ids{exec, input_size};
    thrust::upper_bound(policy, range_bounds + 1, range_bounds + num_ranges + 1,
                        idxs_ptr, idxs_ptr + input_size, range_ids.get_data());
    return range_ids;
}


template <typename LocalIndexType, typename GlobalIndexType>
void build_mapping(
    std::shared_ptr<const DefaultExecutor> exec,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        part,
    const array<GlobalIndexType>& recv_connections,
    array<experimental::distributed::comm_index_type>& remote_part_ids,
    array<LocalIndexType>& remote_local_idxs,
    array<GlobalIndexType>& remote_global_idxs, array<int64>& remote_sizes)
{
    auto part_ids = part->get_part_ids();
    auto num_parts = static_cast<size_type>(part->get_num_parts());
    const auto* range_bounds = part->get_range_bounds();
    const auto* range_starting_indices = part->get_range_starting_indices();
    const auto num_ranges = part->get_num_ranges();
    auto input_size = recv_connections.get_size();

    auto recv_connections_copy = recv_connections;
    auto recv_connections_ptr = recv_connections_copy.get_data();

    auto policy = thrust_policy(exec);

    // precompute the range id of each input element
    auto range_ids = compute_range_ids(exec, part, recv_connections_copy);
    auto it_range_ids = range_ids.get_data();

    // map input to owning part-id
    array<experimental::distributed::comm_index_type> full_remote_part_ids(
        exec, input_size);
    auto it_full_remote_part_ids = full_remote_part_ids.get_data();
    thrust::transform(policy, it_range_ids, it_range_ids + input_size,
                      it_full_remote_part_ids,
                      [part_ids] __host__ __device__(const size_type rid) {
                          return part_ids[rid];
                      });

    // sort by part-id and recv_connection
    auto sort_it = thrust::make_zip_iterator(
        thrust::make_tuple(it_full_remote_part_ids, recv_connections_ptr));
    thrust::sort_by_key(policy, sort_it, sort_it + input_size, it_range_ids);

    auto unique_end = thrust::unique_by_key(policy, sort_it,
                                            sort_it + input_size, it_range_ids);
    auto unique_range_id_end = unique_end.second;
    auto unique_size = thrust::distance(it_range_ids, unique_range_id_end);

    remote_global_idxs.resize_and_reset(unique_size);
    remote_local_idxs.resize_and_reset(unique_size);

    // store unique connections, also map global indices to local
    {
        auto copy_it = thrust::make_zip_iterator(
            thrust::make_tuple(recv_connections_ptr, it_range_ids));
        thrust::copy_n(policy, copy_it, unique_size,
                       thrust::make_transform_output_iterator(
                           thrust::make_zip_iterator(thrust::make_tuple(
                               remote_global_idxs.get_data(),
                               remote_local_idxs.get_data())),
                           transform_output<LocalIndexType, GlobalIndexType>{
                               range_bounds, range_starting_indices}));
    }

    // compute number of connections per part-id
    array<unsigned long long int> full_remote_sizes(exec,
                                                    part->get_num_parts());
    auto recv_sizes_ptr = full_remote_sizes.get_data();
    thrust::fill_n(policy, recv_sizes_ptr, num_parts, 0);
    thrust::for_each_n(policy, it_full_remote_part_ids, unique_size,
                       [recv_sizes_ptr] __device__(const size_type part) {
                           atomic_add(recv_sizes_ptr + part, 1);
                       });

    auto is_neighbor = [] __host__ __device__(const size_type s) {
        return s != 0;
    };
    auto num_neighbors =
        thrust::count_if(policy, recv_sizes_ptr,
                         recv_sizes_ptr + part->get_num_parts(), is_neighbor);

    remote_part_ids.resize_and_reset(num_neighbors);

    remote_sizes.resize_and_reset(num_neighbors);
    {
        auto counting_it = thrust::make_counting_iterator(0);
        auto copy_it = thrust::make_zip_iterator(
            thrust::make_tuple(recv_sizes_ptr, counting_it));
        thrust::copy_if(
            policy, copy_it, copy_it + part->get_num_parts(),
            thrust::make_zip_iterator(thrust::make_tuple(
                remote_sizes.get_data(), remote_part_ids.get_data())),
            [] __host__ __device__(
                const thrust::tuple<unsigned long long int, int>& t) {
                return thrust::get<0>(t) > 0;
            });
    }
}

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_INDEX_MAP_BUILD_MAPPING);


template <typename LocalIndexType, typename GlobalIndexType>
void map_to_local(
    std::shared_ptr<const DefaultExecutor> exec,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        partition,
    const array<experimental::distributed::comm_index_type>& remote_target_ids,
    device_segmented_array<const GlobalIndexType> remote_global_idxs,
    experimental::distributed::comm_index_type rank,
    const array<GlobalIndexType>& global_ids,
    experimental::distributed::index_space is, array<LocalIndexType>& local_ids)
{
    auto part_ids = partition->get_part_ids();
    auto part_sizes = partition->get_part_sizes();
    auto num_parts = static_cast<size_type>(partition->get_num_parts());
    const auto* range_bounds = partition->get_range_bounds();
    const auto* range_starting_indices =
        partition->get_range_starting_indices();
    const auto num_ranges = partition->get_num_ranges();
    auto input_size = global_ids.get_size();
    auto global_ids_it = global_ids.get_const_data();

    auto policy = thrust_policy(exec);

    local_ids.resize_and_reset(input_size);
    auto local_ids_it = local_ids.get_data();

    auto range_ids = compute_range_ids(exec, partition, global_ids);
    auto range_ids_it = range_ids.get_const_data();

    auto map_local =
        [rank, part_ids, range_bounds, range_starting_indices] __device__(
            const thrust::tuple<GlobalIndexType, size_type>& t) {
            auto gid = thrust::get<0>(t);
            auto rid = thrust::get<1>(t);
            auto pid = part_ids[rid];
            return pid == rank
                       ? static_cast<LocalIndexType>(gid - range_bounds[rid]) +
                             range_starting_indices[rid]
                       : invalid_index<LocalIndexType>();
        };

    auto remote_target_ids_ptr = remote_target_ids.get_const_data();
    auto num_target_ids = remote_target_ids.get_size();
    auto remote_global_idxs_ptr = remote_global_idxs.flat_begin;
    auto offsets_ptr = remote_global_idxs.offsets_begin;
    auto map_non_local =
        [num_target_ids, remote_target_ids_ptr, part_ids, offsets_ptr,
         remote_global_idxs_ptr] __device__(const thrust::tuple<GlobalIndexType,
                                                                size_type>& t) {
            auto gid = thrust::get<0>(t);
            auto rid = thrust::get<1>(t);
            auto pid = part_ids[rid];
            auto set_id = binary_search(
                size_type{0}, num_target_ids,
                [=](const auto i) { return remote_target_ids_ptr[i] >= pid; });

            // Set an invalid index, if the part-id could not be found
            if (set_id == num_target_ids) {
                return invalid_index<LocalIndexType>();
            }

            // need to check if *it is actually the current global-id
            // since the global-id might not be registered as connected
            // to this rank
            auto it = binary_search(
                offsets_ptr[set_id],
                offsets_ptr[set_id + 1] - offsets_ptr[set_id],
                [=](const auto i) { return remote_global_idxs_ptr[i] >= gid; });
            return it != offsets_ptr[set_id + 1] &&
                           remote_global_idxs_ptr[it] == gid
                       ? static_cast<LocalIndexType>(it)
                       : invalid_index<LocalIndexType>();
        };

    auto map_combined =
        [part_ids, rank, map_local, map_non_local, part_sizes] __device__(
            const thrust::tuple<GlobalIndexType, size_type>& t) {
            auto gid = thrust::get<0>(t);
            auto rid = thrust::get<1>(t);
            auto pid = part_ids[rid];

            if (pid == rank) {
                return map_local(t);
            } else {
                auto id = map_non_local(t);
                return id == invalid_index<LocalIndexType>()
                           ? id
                           : id + part_sizes[rank];
            }
        };

    auto transform_it = thrust::make_zip_iterator(
        thrust::make_tuple(global_ids_it, range_ids_it));
    if (is == experimental::distributed::index_space::local) {
        thrust::transform(policy, transform_it, transform_it + input_size,
                          local_ids_it, map_local);
    }
    if (is == experimental::distributed::index_space::non_local) {
        thrust::transform(policy, transform_it, transform_it + input_size,
                          local_ids_it, map_non_local);
    }
    if (is == experimental::distributed::index_space::combined) {
        thrust::transform(policy, transform_it, transform_it + input_size,
                          local_ids_it, map_combined);
    }
}

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_INDEX_MAP_MAP_TO_LOCAL);


template <typename LocalIndexType, typename GlobalIndexType>
void map_to_global(
    std::shared_ptr<const DefaultExecutor> exec,
    device_partition<const LocalIndexType, const GlobalIndexType> partition,
    device_segmented_array<const GlobalIndexType> remote_global_idxs,
    experimental::distributed::comm_index_type rank,
    const array<LocalIndexType>& local_idxs,
    experimental::distributed::index_space is,
    array<GlobalIndexType>& global_idxs)
{
    auto range_bounds = partition.offsets_begin;
    auto starting_indices = partition.starting_indices_begin;
    const auto& ranges_by_part = partition.ranges_by_part;
    auto local_idxs_it = local_idxs.get_const_data();
    auto input_size = local_idxs.get_size();

    auto policy = thrust_policy(exec);

    global_idxs.resize_and_reset(local_idxs.get_size());
    auto global_idxs_it = global_idxs.get_data();

    auto map_local = [rank, ranges_by_part, range_bounds, starting_indices,
                      partition] __device__(auto lid) {
        auto local_size =
            static_cast<LocalIndexType>(partition.part_sizes_begin[rank]);

        if (lid < 0 || lid >= local_size) {
            return invalid_index<GlobalIndexType>();
        }

        auto local_ranges = ranges_by_part.get_segment(rank);
        auto local_ranges_size =
            static_cast<int64>(local_ranges.end - local_ranges.begin);

        // the binary search finds the first local range, such that the starting
        // index is larger than lid, thus lid is contained in the local range
        // before that one
        auto local_range_id =
            binary_search(int64(0), local_ranges_size,
                          [=](const auto i) {
                              return starting_indices[local_ranges.begin[i]] >
                                     lid;
                          }) -
            1;
        auto range_id = local_ranges.begin[local_range_id];

        return static_cast<GlobalIndexType>(lid - starting_indices[range_id]) +
               range_bounds[range_id];
    };
    auto map_non_local = [remote_global_idxs] __device__(auto lid) {
        auto remote_size = static_cast<LocalIndexType>(
            remote_global_idxs.flat_end - remote_global_idxs.flat_begin);

        if (lid < 0 || lid >= remote_size) {
            return invalid_index<GlobalIndexType>();
        }

        return remote_global_idxs.flat_begin[lid];
    };
    auto map_combined = [map_local, map_non_local, partition,
                         rank] __device__(auto lid) {
        auto local_size =
            static_cast<LocalIndexType>(partition.part_sizes_begin[rank]);

        if (lid < local_size) {
            return map_local(lid);
        } else {
            return map_non_local(lid - local_size);
        }
    };

    if (is == experimental::distributed::index_space::local) {
        thrust::transform(policy, local_idxs_it, local_idxs_it + input_size,
                          global_idxs_it, map_local);
    }
    if (is == experimental::distributed::index_space::non_local) {
        thrust::transform(policy, local_idxs_it, local_idxs_it + input_size,
                          global_idxs_it, map_non_local);
    }
    if (is == experimental::distributed::index_space::combined) {
        thrust::transform(policy, local_idxs_it, local_idxs_it + input_size,
                          global_idxs_it, map_combined);
    }
}

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_INDEX_MAP_MAP_TO_GLOBAL);


}  // namespace index_map
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
