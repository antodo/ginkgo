// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/test/utils/assertions.hpp"

#include <type_traits>

#include <gtest/gtest.h>

#include <ginkgo/core/base/segmented_array.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace {


class MatricesNear : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    using Sparse = gko::matrix::Csr<>;

    template <typename Type, std::size_t size>
    gko::array<Type> make_view(std::array<Type, size>& array)
    {
        return gko::make_array_view(exec, size, array.data());
    }

    MatricesNear()
        : exec(gko::ReferenceExecutor::create()),
          mtx1(gko::initialize<Mtx>({{1.0, 2.0, 3.0}, {0.0, 4.0, 0.0}}, exec)),
          mtx2(gko::initialize<Mtx>({{1.0, 2.0, 3.0}, {4.0, 0.0, 4.0}}, exec)),
          mtx3(gko::initialize<Mtx>({{1.0, 2.0, 3.0}, {0.0, 4.1, 0.0}}, exec)),
          mtx13_row_ptrs({0, 3, 4}),
          mtx2_row_ptrs({0, 3, 5}),
          mtx13_col_idxs({0, 1, 2, 1}),
          mtx2_col_idxs({0, 1, 2, 0, 2}),
          mtx1_vals({1.0, 2.0, 3.0, 4.0}),
          mtx2_vals({1.0, 2.0, 3.0, 4.0, 4.0}),
          mtx3_vals({1.0, 2.0, 3.0, 4.1})
    {
        mtx1_sp = Sparse::create(exec, mtx1->get_size(), make_view(mtx1_vals),
                                 make_view(mtx13_col_idxs),
                                 make_view(mtx13_row_ptrs));
        mtx2_sp =
            Sparse::create(exec, mtx2->get_size(), make_view(mtx2_vals),
                           make_view(mtx2_col_idxs), make_view(mtx2_row_ptrs));
        mtx3_sp = Sparse::create(exec, mtx3->get_size(), make_view(mtx3_vals),
                                 make_view(mtx13_col_idxs),
                                 make_view(mtx13_row_ptrs));
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx1;
    std::unique_ptr<Mtx> mtx2;
    std::unique_ptr<Mtx> mtx3;
    std::array<Sparse::index_type, 3> mtx13_row_ptrs;
    std::array<Sparse::index_type, 3> mtx2_row_ptrs;
    std::array<Sparse::index_type, 4> mtx13_col_idxs;
    std::array<Sparse::index_type, 5> mtx2_col_idxs;
    std::array<Sparse::value_type, 4> mtx1_vals;
    std::array<Sparse::value_type, 5> mtx2_vals;
    std::array<Sparse::value_type, 4> mtx3_vals;
    std::unique_ptr<Sparse> mtx1_sp;
    std::unique_ptr<Sparse> mtx2_sp;
    std::unique_ptr<Sparse> mtx3_sp;
};


TEST_F(MatricesNear, SucceedsIfSame)
{
    ASSERT_PRED_FORMAT3(gko::test::assertions::matrices_near, mtx1.get(),
                        mtx1.get(), 0.0);
    ASSERT_PRED_FORMAT2(gko::test::assertions::matrices_equal_sparsity,
                        mtx1_sp.get(), mtx1_sp.get());
}


TEST_F(MatricesNear, FailsIfDifferent)
{
    ASSERT_PRED_FORMAT3(!gko::test::assertions::matrices_near, mtx1.get(),
                        mtx2.get(), 0.0);
    ASSERT_PRED_FORMAT2(!gko::test::assertions::matrices_equal_sparsity,
                        mtx1_sp.get(), mtx2_sp.get());
}


TEST_F(MatricesNear, SucceedsIfClose)
{
    ASSERT_PRED_FORMAT3(!gko::test::assertions::matrices_near, mtx1.get(),
                        mtx3.get(), 0.0);
    ASSERT_PRED_FORMAT3(gko::test::assertions::matrices_near, mtx1.get(),
                        mtx3.get(), 0.1);
    ASSERT_PRED_FORMAT2(gko::test::assertions::matrices_equal_sparsity,
                        mtx1_sp.get(), mtx3_sp.get());
}


TEST_F(MatricesNear, CanUseShortNotation)
{
    GKO_EXPECT_MTX_NEAR(mtx1, mtx1, 0.0);
    GKO_ASSERT_MTX_NEAR(mtx1, mtx3, 0.1);
    GKO_EXPECT_MTX_EQ_SPARSITY(mtx1_sp, mtx3_sp);
    GKO_ASSERT_MTX_EQ_SPARSITY(mtx1_sp, mtx3_sp);
}


TEST_F(MatricesNear, CanPassInitializerList)
{
    GKO_EXPECT_MTX_NEAR(mtx1, l({{1.0, 2.0, 3.0}, {0.0, 4.0, 0.0}}), 0.0);
    GKO_ASSERT_MTX_NEAR(mtx1, l({{1.0, 2.0, 3.0}, {0.0, 4.0, 0.0}}), 0.0);
}


TEST(BiggestValueType, SameNonComplex)
{
    using T1 = float;
    using T2 = float;
    using result =
        gko::test::assertions::detail::biggest_valuetype<T1, T2>::type;

    bool is_float = std::is_same<result, float>::value;
    ASSERT_TRUE(is_float);
}


TEST(BiggestValueType, BetweenNonComplex)
{
    using T1 = float;
    using T2 = double;
    using result =
        gko::test::assertions::detail::biggest_valuetype<T1, T2>::type;

    bool is_double = std::is_same<result, double>::value;
    ASSERT_TRUE(is_double);
}


TEST(BiggestValueType, WithSameComplex)
{
    using T1 = std::complex<float>;
    using T2 = std::complex<float>;
    using result =
        gko::test::assertions::detail::biggest_valuetype<T1, T2>::type;

    bool is_cpx_float = std::is_same<result, std::complex<float>>::value;
    ASSERT_TRUE(is_cpx_float);
}


TEST(BiggestValueType, WithAComplex)
{
    using T1 = std::complex<float>;
    using T2 = double;
    using result =
        gko::test::assertions::detail::biggest_valuetype<T1, T2>::type;

    bool is_cpx_double = std::is_same<result, std::complex<double>>::value;
    ASSERT_TRUE(is_cpx_double);
}


TEST(BiggestValueType, WithBothComplex)
{
    using T1 = std::complex<float>;
    using T2 = std::complex<double>;
    using result =
        gko::test::assertions::detail::biggest_valuetype<T1, T2>::type;

    bool is_cpx_double = std::is_same<result, std::complex<double>>::value;
    ASSERT_TRUE(is_cpx_double);
}


class ArraysNear : public ::testing::Test {
protected:
    using ValueArray = gko::array<double>;

    ArraysNear()
        : exec(gko::ReferenceExecutor::create()),
          arr1(exec, {1.0, 2.0, 3.0, 4.2}),
          arr2(exec, {1.0, 2.0, 3.0, 4.2}),
          arr3(exec, {1.0, 2.0, 3.00001, 4.2}),
          arr4(exec, {1.0, 2.0, 3.0, 4.3})
    {}

    std::shared_ptr<const gko::Executor> exec;
    ValueArray arr1;
    ValueArray arr2;
    ValueArray arr3;
    ValueArray arr4;
};


TEST_F(ArraysNear, SucceedsIfEqual)
{
    ASSERT_PRED_FORMAT3(gko::test::assertions::array_near, arr1, arr2, 0.0);
}


TEST_F(ArraysNear, FailsIfDifferent)
{
    ASSERT_PRED_FORMAT3(!gko::test::assertions::array_near, arr1, arr4, 0.0);
}


TEST_F(ArraysNear, SucceedsIfClose)
{
    ASSERT_PRED_FORMAT3(!gko::test::assertions::array_near, arr1, arr3, 0.0);
    ASSERT_PRED_FORMAT3(gko::test::assertions::array_near, arr1, arr3, 0.0001);
}


TEST_F(ArraysNear, CanUseShortNotation)
{
    GKO_EXPECT_ARRAY_NEAR(arr1, arr2, 0.0);
    GKO_ASSERT_ARRAY_NEAR(arr1, arr3, 0.001);
}


class SegmentedArraysEqual : public ::testing::Test {
protected:
    using array = gko::array<double>;
    using iarray = gko::array<gko::int64>;
    using segmented_array = gko::segmented_array<double>;

    std::shared_ptr<gko::Executor> exec = gko::ReferenceExecutor::create();

    segmented_array arr1 = segmented_array::create_from_sizes(
        array{exec, {1, 2, 3, 4, 5}}, iarray{exec, {2, 1, 2}});
    segmented_array arr2 = segmented_array::create_from_sizes(
        array{exec, {1, 2, 3, 4, 5}}, iarray{exec, {2, 1, 2}});
    segmented_array arr3 = segmented_array::create_from_sizes(
        array{exec, {1, 2, 3, 5, 6}}, iarray{exec, {2, 1, 2}});
    segmented_array arr4 = segmented_array::create_from_sizes(
        array{exec, {1, 2, 3, 4, 5}}, iarray{exec, {3, 2}});
    segmented_array arr5 = segmented_array::create_from_sizes(
        array{exec, {1, 2, 3, 4, 5}}, iarray{exec, {1, 2, 2}});
};


TEST_F(SegmentedArraysEqual, SucceedsIfEqual)
{
    GKO_ASSERT_SEGMENTED_ARRAY_EQ(arr1, arr2);
}


TEST_F(SegmentedArraysEqual, FailsIfValuesDifferent)
{
    ASSERT_PRED_FORMAT2(!::gko::test::assertions::segmented_array_equal, arr1,
                        arr3);
}


TEST_F(SegmentedArraysEqual, FailsIfOffsetsDifferent1)
{
    ASSERT_PRED_FORMAT2(!::gko::test::assertions::segmented_array_equal, arr1,
                        arr4);
}


TEST_F(SegmentedArraysEqual, FailsIfOffsetsDifferent2)
{
    ASSERT_PRED_FORMAT2(!::gko::test::assertions::segmented_array_equal, arr1,
                        arr5);
}


}  // namespace
