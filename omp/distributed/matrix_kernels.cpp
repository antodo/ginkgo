/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include "core/distributed/matrix_kernels.hpp"


#include <omp.h>


#include <ginkgo/core/base/exception_helpers.hpp>


#include "core/base/allocator.hpp"
#include "core/base/device_matrix_data_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace distributed_matrix {


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void build_diag_offdiag(
    std::shared_ptr<const DefaultExecutor> exec,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    const distributed::Partition<LocalIndexType, GlobalIndexType>* partition,
    comm_index_type local_part,
    device_matrix_data<ValueType, LocalIndexType>& diag_data,
    device_matrix_data<ValueType, LocalIndexType>& offdiag_data,
    Array<LocalIndexType>& local_gather_idxs, comm_index_type* recv_offsets,
    Array<GlobalIndexType>& local_to_global_ghost)
{
    auto input_row_idxs = input.get_const_row_idxs();
    auto input_col_idxs = input.get_const_col_idxs();
    auto input_vals = input.get_const_values();
    using range_index_type = GlobalIndexType;
    using global_nonzero = matrix_data_entry<ValueType, GlobalIndexType>;
    using local_nonzero = matrix_data_entry<ValueType, LocalIndexType>;
    auto range_bounds = partition->get_range_bounds();
    auto part_ids = partition->get_part_ids();
    auto range_starting_indices = partition->get_range_starting_indices();
    auto num_parts = partition->get_num_parts();
    auto num_ranges = partition->get_num_ranges();
    size_type row_range_id_hint = 0;
    size_type col_range_id_hint = 0;
    // zero recv_offsets values
    std::fill_n(recv_offsets, num_parts + 1, comm_index_type{});

    auto find_range = [&](GlobalIndexType idx, size_type hint) {
        if (range_bounds[hint] <= idx && idx < range_bounds[hint + 1]) {
            return hint;
        } else {
            auto it = std::upper_bound(range_bounds + 1,
                                       range_bounds + num_ranges + 1, idx);
            return static_cast<size_type>(std::distance(range_bounds + 1, it));
        }
    };
    auto map_to_local = [&](GlobalIndexType idx,
                            size_type range_id) -> LocalIndexType {
        return static_cast<LocalIndexType>(idx - range_bounds[range_id]) +
               range_starting_indices[range_id];
    };

    // store offdiagonal columns and their range indices
    std::map<GlobalIndexType, range_index_type> offdiag_cols;
    // store offdiagonal entries with global column idxs
    std::vector<global_nonzero> global_offdiag_entries;
    std::vector<local_nonzero> diag_entries;

    auto num_threads = static_cast<size_type>(omp_get_max_threads());
    auto num_input = input.get_num_elems();
    auto size_per_thread = (num_input + num_threads - 1) / num_threads;
    std::vector<size_type> diag_entry_offsets(num_threads, 0);
    std::vector<size_type> offdiag_entry_offsets(num_threads, 0);

#pragma omp parallel num_threads(num_threads)
    {
        std::unordered_map<GlobalIndexType, range_index_type>
            thread_offdiag_cols;
        std::vector<global_nonzero> thread_offdiag_entries;
        std::vector<local_nonzero> thread_diag_entries;
        std::vector<comm_index_type> thread_recv_sizes;
        auto thread_id = omp_get_thread_num();
        auto thread_begin = thread_id * size_per_thread;
        auto thread_end = std::min(thread_begin + size_per_thread, num_input);
        // separate diagonal and off-diagonal entries for our input chunk
        for (auto i = thread_begin; i < thread_end; ++i) {
            const auto global_row = input_row_idxs[i];
            const auto global_col = input_col_idxs[i];
            const auto value = input_vals[i];
            auto row_range_id = find_range(global_row, row_range_id_hint);
            row_range_id_hint = row_range_id;
            // skip non-local rows
            if (part_ids[row_range_id] == local_part) {
                // map to part-local indices
                auto local_row = map_to_local(global_row, row_range_id);

                auto col_range_id = find_range(global_col, col_range_id_hint);
                col_range_id_hint = col_range_id;
                if (part_ids[col_range_id] == local_part) {
                    // store diagonal entry
                    auto local_col = map_to_local(global_col, col_range_id);
                    thread_diag_entries.emplace_back(local_row, local_col,
                                                     value);
                } else {
                    thread_offdiag_cols.emplace(global_col, col_range_id);
                    thread_offdiag_entries.emplace_back(local_row, global_col,
                                                        value);
                }
            }
        }
        diag_entry_offsets[thread_id] = thread_diag_entries.size();
        offdiag_entry_offsets[thread_id] = thread_offdiag_entries.size();

#pragma omp critical
        {
            // collect global off-diagonal columns
            offdiag_cols.insert(thread_offdiag_cols.begin(),
                                thread_offdiag_cols.end());
        }
#pragma omp barrier
#pragma omp single
        {
            // assign output ranges to the individual threads
            size_type diag{};
            size_type offdiag{};
            for (size_type thread = 0; thread < num_threads; ++thread) {
                auto size_diag = diag_entry_offsets[thread];
                auto size_offdiag = offdiag_entry_offsets[thread];
                diag_entry_offsets[thread] = diag;
                offdiag_entry_offsets[thread] = offdiag;
                diag += size_diag;
                offdiag += size_offdiag;
            }
            diag_entries.resize(diag);
            global_offdiag_entries.resize(offdiag);
        }
        // write back the local data to the output ranges
        auto diag = diag_entry_offsets[thread_id];
        auto offdiag = offdiag_entry_offsets[thread_id];
        for (auto& entry : thread_diag_entries) {
            diag_entries[diag] = entry;
            diag++;
        }
        for (auto& entry : thread_offdiag_entries) {
            global_offdiag_entries[offdiag] = entry;
            offdiag++;
        }
    }
    // store diagonal data to output
    const auto num_diag_elems =
        static_cast<size_type>(partition->get_part_size(local_part));
    diag_data.resize_and_reset(gko::dim<2>{num_diag_elems, num_diag_elems},
                               diag_entries.size());
    components::aos_to_soa(exec,
                           Array<local_nonzero>::view(exec, diag_entries.size(),
                                                      diag_entries.data()),
                           diag_data);
    // count off-diagonal columns per part
    for (auto entry : offdiag_cols) {
        auto col_range_id = entry.second;
        recv_offsets[part_ids[col_range_id]]++;
    }
    components::prefix_sum(exec, recv_offsets, num_parts + 1);
    const auto num_ghost_elems =
        static_cast<size_type>(recv_offsets[num_parts]);
    local_gather_idxs.resize_and_reset(num_ghost_elems);
    std::unordered_map<GlobalIndexType, LocalIndexType> offdiag_global_to_local;
    // collect and renumber offdiagonal columns
    for (auto entry : offdiag_cols) {
        auto range = entry.second;
        auto range_begin = range_bounds[range];
        auto starting_index = range_starting_indices[range];
        auto part = part_ids[range];
        auto idx = recv_offsets[part];
        local_gather_idxs.get_data()[idx] = static_cast<comm_index_type>(
            entry.first - range_begin + starting_index);
        offdiag_global_to_local[entry.first] = idx;
        ++recv_offsets[part];
    }
    // build local-to-global map for offdiag columns
    local_to_global_ghost.resize_and_reset(num_ghost_elems);
    local_to_global_ghost.fill(invalid_index<GlobalIndexType>());
    for (const auto& key_value : offdiag_global_to_local) {
        const auto global_idx = key_value.first;
        const auto local_idx = key_value.second;
        local_to_global_ghost.get_data()[local_idx] = global_idx;
    }
    // shift recv_offsets to the back, insert 0 in front again
    LocalIndexType local_prev{};
    for (size_type i = 0; i <= num_parts; i++) {
        recv_offsets[i] = std::exchange(local_prev, recv_offsets[i]);
    }
    // map off-diag values to local column indices
    offdiag_data.resize_and_reset(gko::dim<2>{num_diag_elems, num_ghost_elems},
                                  global_offdiag_entries.size());
#pragma omp for
    for (size_type i = 0; i < global_offdiag_entries.size(); i++) {
        auto global = global_offdiag_entries[i];
        offdiag_data.get_row_idxs()[i] =
            static_cast<LocalIndexType>(global.row);
        offdiag_data.get_col_idxs()[i] = offdiag_global_to_local[global.column];
        offdiag_data.get_values()[i] = global.value;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_BUILD_DIAG_OFFDIAG);


template <typename SourceType, typename TargetType>
void map_to_global_idxs(std::shared_ptr<const DefaultExecutor> exec,
                        const SourceType* input, size_t n, TargetType* output,
                        const TargetType* map) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_MAP_TO_GLOBAL_IDXS);


}  // namespace distributed_matrix
}  // namespace omp
}  // namespace kernels
}  // namespace gko
