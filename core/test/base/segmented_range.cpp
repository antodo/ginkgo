// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <iterator>
#include <numeric>
#include <vector>


#include <gtest/gtest.h>


#include "core/base/segmented_range.hpp"


TEST(SegmentedRange, Works)
{
    std::vector<int> begins{3, 1, 4, 9};
    std::vector<int> ends{3, 10, 6, 10};
    std::vector<std::vector<int>> result_indices(begins.size());
    gko::segmented_range<int> range{begins.data(), ends.data(),
                                    static_cast<int>(begins.size())};

    for (auto row : gko::irange<int>(begins.size())) {
        for (auto nz : range[row]) {
            result_indices[row].push_back(nz);
        }
    }

    ASSERT_EQ(result_indices,
              std::vector<std::vector<int>>(
                  {{}, {1, 2, 3, 4, 5, 6, 7, 8, 9}, {4, 5}, {9}}));
}


TEST(SegmentedValueRange, WorksByIndex)
{
    std::vector<int> begins{3, 1, 4, 9};
    std::vector<int> ends{3, 10, 6, 10};
    std::vector<int> values(ends.back());
    std::iota(values.begin(), values.end(), 1);
    std::vector<std::vector<int>> result_values(begins.size());
    gko::segmented_value_range<int, std::vector<int>::iterator> range{
        begins.data(), ends.data(), values.begin(),
        static_cast<int>(begins.size())};

    for (auto row : gko::irange<int>(begins.size())) {
        for (auto nz : range[row]) {
            result_values[row].push_back(nz);
        }
    }

    ASSERT_EQ(result_values,
              std::vector<std::vector<int>>(
                  {{}, {2, 3, 4, 5, 6, 7, 8, 9, 10}, {5, 6}, {10}}));
}


TEST(SegmentedValueRange, WorksByRangeFor)
{
    std::vector<int> begins{3, 1, 4, 9};
    std::vector<int> ends{3, 10, 6, 10};
    std::vector<int> values(ends.back());
    std::iota(values.begin(), values.end(), 1);
    std::vector<std::vector<int>> result_values(begins.size());
    gko::segmented_value_range<int, std::vector<int>::iterator> range{
        begins.data(), ends.data(), values.begin(),
        static_cast<int>(begins.size())};

    for (auto [row, segment] : range) {
        for (auto nz : segment) {
            result_values[row].push_back(nz);
        }
    }

    ASSERT_EQ(result_values,
              std::vector<std::vector<int>>(
                  {{}, {2, 3, 4, 5, 6, 7, 8, 9, 10}, {5, 6}, {10}}));
}


TEST(SegmentedEnumeratedValueRange, WorksByIndex)
{
    using gko::get;
    std::vector<int> begins{3, 1, 4, 9};
    std::vector<int> ends{3, 10, 6, 10};
    std::vector<int> values(ends.back());
    std::iota(values.begin(), values.end(), 1);
    std::vector<std::vector<int>> result_values(begins.size());
    std::vector<std::vector<int>> result_indices(begins.size());
    gko::segmented_value_range<int, std::vector<int>::iterator> range{
        begins.data(), ends.data(), values.begin(),
        static_cast<int>(begins.size())};

    for (auto row : gko::irange<int>(begins.size())) {
        for (auto tuple : range.enumerated()[row]) {
            result_indices[row].push_back(get<0>(tuple));
            result_values[row].push_back(get<1>(tuple));
        }
    }

    ASSERT_EQ(result_indices,
              std::vector<std::vector<int>>(
                  {{}, {1, 2, 3, 4, 5, 6, 7, 8, 9}, {4, 5}, {9}}));
    ASSERT_EQ(result_values,
              std::vector<std::vector<int>>(
                  {{}, {2, 3, 4, 5, 6, 7, 8, 9, 10}, {5, 6}, {10}}));
}


TEST(SegmentedEnumeratedValueRange, WorksByRangeFor)
{
    std::vector<int> begins{3, 1, 4, 9};
    std::vector<int> ends{3, 10, 6, 10};
    std::vector<int> values(ends.back());
    std::iota(values.begin(), values.end(), 1);
    std::vector<std::vector<int>> result_values(begins.size());
    std::vector<std::vector<int>> result_indices(begins.size());
    gko::segmented_value_range<int, std::vector<int>::iterator> range{
        begins.data(), ends.data(), values.begin(),
        static_cast<int>(begins.size())};
    auto enumerated_range = range.enumerated();

    for (auto [row, segment] : enumerated_range) {
        for (auto [index, value] : segment) {
            result_indices[row].push_back(index);
            result_values[row].push_back(value);
        }
    }

    ASSERT_EQ(result_indices,
              std::vector<std::vector<int>>(
                  {{}, {1, 2, 3, 4, 5, 6, 7, 8, 9}, {4, 5}, {9}}));
    ASSERT_EQ(result_values,
              std::vector<std::vector<int>>(
                  {{}, {2, 3, 4, 5, 6, 7, 8, 9, 10}, {5, 6}, {10}}));
}
