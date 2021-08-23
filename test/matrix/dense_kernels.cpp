/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/matrix/dense.hpp>


#include <algorithm>
#include <numeric>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/components/fill_array.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


namespace {


class Dense : public ::testing::Test {
protected:
    using itype = int;
#if GINKGO_COMMON_SINGLE_MODE
    using vtype = float;
#else
    using vtype = double;
#endif
    using Mtx = gko::matrix::Dense<vtype>;
    using MixedMtx = gko::matrix::Dense<gko::next_precision<vtype>>;
    using NormVector = gko::matrix::Dense<gko::remove_complex<vtype>>;
    using Arr = gko::Array<itype>;
    using ComplexMtx = gko::matrix::Dense<std::complex<vtype>>;
    using Diagonal = gko::matrix::Diagonal<vtype>;
    using MixedComplexMtx =
        gko::matrix::Dense<gko::next_precision<std::complex<vtype>>>;

    Dense() : rand_engine(15) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        init_executor(ref, exec);
    }

    void TearDown()
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(0.0, 1.0), rand_engine, ref);
    }

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols,
                                     int min_nnz_row)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(min_nnz_row, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_vector_data(gko::size_type num_vecs,
                            bool different_alpha = false)
    {
        x = gen_mtx<Mtx>(1000, num_vecs);
        y = gen_mtx<Mtx>(1000, num_vecs);
        if (different_alpha) {
            alpha = gen_mtx<Mtx>(1, num_vecs);
        } else {
            alpha = gko::initialize<Mtx>({2.0}, ref);
        }
        dx = Mtx::create(exec);
        dx->copy_from(x.get());
        dy = Mtx::create(exec);
        dy->copy_from(y.get());
        dalpha = Mtx::create(exec);
        dalpha->copy_from(alpha.get());
        expected = Mtx::create(ref, gko::dim<2>{1, num_vecs});
        dresult = Mtx::create(exec, gko::dim<2>{1, num_vecs});
    }

    void set_up_apply_data()
    {
        x = gen_mtx<Mtx>(40, 25);
        c_x = gen_mtx<ComplexMtx>(40, 25);
        y = gen_mtx<Mtx>(25, 35);
        expected = gen_mtx<Mtx>(40, 35);
        alpha = gko::initialize<Mtx>({2.0}, ref);
        beta = gko::initialize<Mtx>({-1.0}, ref);
        square = gen_mtx<Mtx>(x->get_size()[0], x->get_size()[0]);
        dx = Mtx::create(exec);
        dx->copy_from(x.get());
        dc_x = ComplexMtx::create(exec);
        dc_x->copy_from(c_x.get());
        dy = Mtx::create(exec);
        dy->copy_from(y.get());
        dresult = Mtx::create(exec);
        dresult->copy_from(expected.get());
        dalpha = Mtx::create(exec);
        dalpha->copy_from(alpha.get());
        dbeta = Mtx::create(exec);
        dbeta->copy_from(beta.get());
        dsquare = Mtx::create(exec);
        dsquare->copy_from(square.get());

        std::vector<int> tmp(x->get_size()[0], 0);
        auto rng = std::default_random_engine{};
        std::iota(tmp.begin(), tmp.end(), 0);
        std::shuffle(tmp.begin(), tmp.end(), rng);
        std::vector<int> tmp2(x->get_size()[1], 0);
        std::iota(tmp2.begin(), tmp2.end(), 0);
        std::shuffle(tmp2.begin(), tmp2.end(), rng);
        std::vector<int> tmp3(x->get_size()[0] / 10);
        std::uniform_int_distribution<int> row_dist(0, x->get_size()[0] - 1);
        for (auto &i : tmp3) {
            i = row_dist(rng);
        }
        rpermute_idxs =
            std::unique_ptr<Arr>(new Arr{ref, tmp.begin(), tmp.end()});
        cpermute_idxs =
            std::unique_ptr<Arr>(new Arr{ref, tmp2.begin(), tmp2.end()});
        rgather_idxs =
            std::unique_ptr<Arr>(new Arr{ref, tmp3.begin(), tmp3.end()});
    }

    template <typename ConvertedType, typename InputType>
    std::unique_ptr<ConvertedType> convert(InputType &&input)
    {
        auto result = ConvertedType::create(input->get_executor());
        input->convert_to(result.get());
        return result;
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;

    std::ranlux48 rand_engine;

    std::unique_ptr<Mtx> x;
    std::unique_ptr<ComplexMtx> c_x;
    std::unique_ptr<Mtx> y;
    std::unique_ptr<Mtx> alpha;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Mtx> expected;
    std::unique_ptr<Mtx> square;
    std::unique_ptr<Mtx> dresult;
    std::unique_ptr<Mtx> dx;
    std::unique_ptr<ComplexMtx> dc_x;
    std::unique_ptr<Mtx> dy;
    std::unique_ptr<Mtx> dalpha;
    std::unique_ptr<Mtx> dbeta;
    std::unique_ptr<Mtx> dsquare;
    std::unique_ptr<Arr> rpermute_idxs;
    std::unique_ptr<Arr> cpermute_idxs;
    std::unique_ptr<Arr> rgather_idxs;
};


TEST_F(Dense, CopyRespectsStride)
{
    set_up_vector_data(3);
    auto stride = dx->get_size()[1] + 1;
    auto result = Mtx::create(exec, dx->get_size(), stride);
    vtype val = 1234567.0;
    auto original_data = result->get_values();
    auto padding_ptr = original_data + dx->get_size()[1];
    exec->copy_from(ref.get(), 1, &val, padding_ptr);

    dx->convert_to(result.get());

    GKO_ASSERT_MTX_NEAR(result, dx, 0);
    ASSERT_EQ(result->get_stride(), stride);
    ASSERT_EQ(exec->copy_val_to_host(padding_ptr), val);
    ASSERT_EQ(result->get_values(), original_data);
}


TEST_F(Dense, FillIsEquivalentToRef)
{
    set_up_vector_data(3);
    auto result = Mtx::create(ref);

    x->fill(42);
    dx->fill(42);
    result->copy_from(dx.get());

    GKO_ASSERT_MTX_NEAR(result, x, 0);
}


TEST_F(Dense, StridedFillIsEquivalentToRef)
{
    using T = vtype;
    auto x = gko::initialize<gko::matrix::Dense<T>>(
        4, {I<T>{1.0, 2.0}, I<T>{3.0, 4.0}, I<T>{5.0, 6.0}}, ref);
    auto dx = gko::initialize<gko::matrix::Dense<T>>(
        4, {I<T>{1.0, 2.0}, I<T>{3.0, 4.0}, I<T>{5.0, 6.0}}, exec);
    auto result = Mtx::create(ref);

    x->fill(42);
    dx->fill(42);
    result->copy_from(dx.get());

    GKO_ASSERT_MTX_NEAR(result, x, 0);
}


TEST_F(Dense, SingleVectorScaleIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    auto result = Mtx::create(ref);
    result->copy_from(dx.get());
    GKO_ASSERT_MTX_NEAR(result, x, r<vtype>::value);
}


TEST_F(Dense, SingleVectorInvScaleIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->inv_scale(alpha.get());
    dx->inv_scale(dalpha.get());

    auto result = Mtx::create(ref);
    result->copy_from(dx.get());
    GKO_ASSERT_MTX_NEAR(result, x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorScaleIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    GKO_ASSERT_MTX_NEAR(dx, x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorInvScaleIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->inv_scale(alpha.get());
    dx->inv_scale(dalpha.get());

    GKO_ASSERT_MTX_NEAR(dx, x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorScaleWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    GKO_ASSERT_MTX_NEAR(dx, x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorInvScaleWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    x->inv_scale(alpha.get());
    dx->inv_scale(dalpha.get());

    GKO_ASSERT_MTX_NEAR(dx, x, r<vtype>::value);
}


TEST_F(Dense, SingleVectorAddScaledIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_MTX_NEAR(dx, x, r<vtype>::value);
}


TEST_F(Dense, SingleVectorSubtractScaledIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->sub_scaled(alpha.get(), y.get());
    dx->sub_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_MTX_NEAR(dx, x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorAddScaledIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_MTX_NEAR(dx, x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorSubtractScaledIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->sub_scaled(alpha.get(), y.get());
    dx->sub_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_MTX_NEAR(dx, x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorAddScaledWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_MTX_NEAR(dx, x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorSubtractScaledWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->sub_scaled(alpha.get(), y.get());
    dx->sub_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_MTX_NEAR(dx, x, r<vtype>::value);
}


TEST_F(Dense, AddsScaledDiagIsEquivalentToRef)
{
    auto mat = gen_mtx<Mtx>(532, 532);
    gko::Array<Mtx::value_type> diag_values(this->ref, 532);
    gko::kernels::reference::components::fill_array(
        this->ref, diag_values.get_data(), 532, Mtx::value_type{2.0});
    auto diag = gko::matrix::Diagonal<Mtx::value_type>::create(this->ref, 532,
                                                               diag_values);
    auto alpha = gko::initialize<Mtx>({2.0}, this->ref);
    auto dmat = Mtx::create(this->exec);
    dmat->copy_from(mat.get());
    auto ddiag = gko::matrix::Diagonal<Mtx::value_type>::create(this->exec);
    ddiag->copy_from(diag.get());
    auto dalpha = Mtx::create(this->exec);
    dalpha->copy_from(alpha.get());

    mat->add_scaled(alpha.get(), diag.get());
    dmat->add_scaled(dalpha.get(), ddiag.get());

    GKO_ASSERT_MTX_NEAR(mat, dmat, r<vtype>::value);
}


TEST_F(Dense, SubtractScaledDiagIsEquivalentToRef)
{
    auto mat = gen_mtx<Mtx>(532, 532);
    gko::Array<Mtx::value_type> diag_values(this->ref, 532);
    gko::kernels::reference::components::fill_array(
        this->ref, diag_values.get_data(), 532, Mtx::value_type{2.0});
    auto diag = gko::matrix::Diagonal<Mtx::value_type>::create(this->ref, 532,
                                                               diag_values);
    auto alpha = gko::initialize<Mtx>({2.0}, this->ref);
    auto dmat = Mtx::create(this->exec);
    dmat->copy_from(mat.get());
    auto ddiag = gko::matrix::Diagonal<Mtx::value_type>::create(this->exec);
    ddiag->copy_from(diag.get());
    auto dalpha = Mtx::create(this->exec);
    dalpha->copy_from(alpha.get());

    mat->sub_scaled(alpha.get(), diag.get());
    dmat->sub_scaled(dalpha.get(), ddiag.get());

    GKO_ASSERT_MTX_NEAR(mat, dmat, r<vtype>::value);
}


TEST_F(Dense, CanGatherRows)
{
    set_up_apply_data();

    auto r_gather = x->row_gather(rgather_idxs.get());
    auto dr_gather = dx->row_gather(rgather_idxs.get());

    GKO_ASSERT_MTX_NEAR(r_gather.get(), dr_gather.get(), 0);
}


TEST_F(Dense, CanGatherRowsIntoDenseCrossExecutor)
{
    set_up_apply_data();
    auto row_span = gko::span{0, x->get_size()[0]};
    auto col_span = gko::span{0, x->get_size()[1] - 2};
    auto sub_x = x->create_submatrix(row_span, col_span);
    auto sub_dx = dx->create_submatrix(row_span, col_span);
    auto gather_size =
        gko::dim<2>{rgather_idxs->get_num_elems(), sub_x->get_size()[1]};
    auto r_gather = Mtx::create(ref, gather_size);
    // test make_temporary_clone and non-default stride
    auto dr_gather = Mtx::create(ref, gather_size, sub_x->get_size()[1] + 2);

    sub_x->row_gather(rgather_idxs.get(), r_gather.get());
    sub_dx->row_gather(rgather_idxs.get(), dr_gather.get());

    GKO_ASSERT_MTX_NEAR(r_gather.get(), dr_gather.get(), 0);
}


TEST_F(Dense, IsPermutable)
{
    set_up_apply_data();

    auto permuted = square->permute(rpermute_idxs.get());
    auto dpermuted = dsquare->permute(rpermute_idxs.get());

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(permuted.get()),
                        static_cast<Mtx *>(dpermuted.get()), 0);
}


TEST_F(Dense, IsPermutableIntoDenseCrossExecutor)
{
    set_up_apply_data();
    auto permuted = Mtx::create(ref, square->get_size());
    // test make_temporary_clone and non-default stride
    auto dpermuted =
        Mtx::create(ref, square->get_size(), square->get_size()[1] + 2);

    square->permute(rpermute_idxs.get(), permuted.get());
    dsquare->permute(rpermute_idxs.get(), dpermuted.get());

    GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
}


TEST_F(Dense, IsInversePermutable)
{
    set_up_apply_data();

    auto permuted = square->inverse_permute(rpermute_idxs.get());
    auto dpermuted = dsquare->inverse_permute(rpermute_idxs.get());

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(permuted.get()),
                        static_cast<Mtx *>(dpermuted.get()), 0);
}


TEST_F(Dense, IsInversePermutableIntoDenseCrossExecutor)
{
    set_up_apply_data();
    auto permuted = Mtx::create(ref, square->get_size());
    // test make_temporary_clone and non-default stride
    auto dpermuted =
        Mtx::create(ref, square->get_size(), square->get_size()[1] + 2);

    square->inverse_permute(rpermute_idxs.get(), permuted.get());
    dsquare->inverse_permute(rpermute_idxs.get(), dpermuted.get());

    GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
}


TEST_F(Dense, IsRowPermutable)
{
    set_up_apply_data();

    auto r_permute = x->row_permute(rpermute_idxs.get());
    auto dr_permute = dx->row_permute(rpermute_idxs.get());

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(r_permute.get()),
                        static_cast<Mtx *>(dr_permute.get()), 0);
}


TEST_F(Dense, IsRowPermutableIntoDenseCrossExecutor)
{
    set_up_apply_data();
    auto permuted = Mtx::create(ref, x->get_size());
    // test make_temporary_clone and non-default stride
    auto dpermuted = Mtx::create(ref, x->get_size(), x->get_size()[1] + 2);

    x->row_permute(rpermute_idxs.get(), permuted.get());
    dx->row_permute(rpermute_idxs.get(), dpermuted.get());

    GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
}


TEST_F(Dense, IsColPermutable)
{
    set_up_apply_data();

    auto c_permute = x->column_permute(cpermute_idxs.get());
    auto dc_permute = dx->column_permute(cpermute_idxs.get());

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(c_permute.get()),
                        static_cast<Mtx *>(dc_permute.get()), 0);
}


TEST_F(Dense, IsColPermutableIntoDenseCrossExecutor)
{
    set_up_apply_data();
    auto permuted = Mtx::create(ref, x->get_size());
    // test make_temporary_clone and non-default stride
    auto dpermuted = Mtx::create(ref, x->get_size(), x->get_size()[1] + 2);

    x->column_permute(cpermute_idxs.get(), permuted.get());
    dx->column_permute(cpermute_idxs.get(), dpermuted.get());

    GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
}


TEST_F(Dense, IsInverseRowPermutable)
{
    set_up_apply_data();

    auto inverse_r_permute = x->inverse_row_permute(rpermute_idxs.get());
    auto d_inverse_r_permute = dx->inverse_row_permute(rpermute_idxs.get());

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(inverse_r_permute.get()),
                        static_cast<Mtx *>(d_inverse_r_permute.get()), 0);
}


TEST_F(Dense, IsInverseRowPermutableIntoDenseCrossExecutor)
{
    set_up_apply_data();
    auto permuted = Mtx::create(ref, x->get_size());
    // test make_temporary_clone and non-default stride
    auto dpermuted = Mtx::create(ref, x->get_size(), x->get_size()[1] + 2);

    x->inverse_row_permute(rpermute_idxs.get(), permuted.get());
    dx->inverse_row_permute(rpermute_idxs.get(), dpermuted.get());

    GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
}


TEST_F(Dense, IsInverseColPermutable)
{
    set_up_apply_data();

    auto inverse_c_permute = x->inverse_column_permute(cpermute_idxs.get());
    auto d_inverse_c_permute = dx->inverse_column_permute(cpermute_idxs.get());

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(inverse_c_permute.get()),
                        static_cast<Mtx *>(d_inverse_c_permute.get()), 0);
}


TEST_F(Dense, IsInverseColPermutableIntoDenseCrossExecutor)
{
    set_up_apply_data();
    auto permuted = Mtx::create(ref, x->get_size());
    // test make_temporary_clone and non-default stride
    auto dpermuted = Mtx::create(ref, x->get_size(), x->get_size()[1] + 2);

    x->inverse_column_permute(cpermute_idxs.get(), permuted.get());
    dx->inverse_column_permute(cpermute_idxs.get(), dpermuted.get());

    GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
}


TEST_F(Dense, ExtractDiagonalOnTallSkinnyIsEquivalentToRef)
{
    set_up_apply_data();

    auto diag = x->extract_diagonal();
    auto ddiag = dx->extract_diagonal();

    GKO_ASSERT_MTX_NEAR(diag.get(), ddiag.get(), 0);
}


TEST_F(Dense, ExtractDiagonalOnTallSkinnyIntoDenseCrossExecutor)
{
    set_up_apply_data();
    auto diag = Diagonal::create(ref, x->get_size()[1]);
    // test make_temporary_clone
    auto ddiag = Diagonal::create(ref, x->get_size()[1]);

    x->extract_diagonal(diag.get());
    dx->extract_diagonal(ddiag.get());

    GKO_ASSERT_MTX_NEAR(diag.get(), ddiag.get(), 0);
}


TEST_F(Dense, ExtractDiagonalOnShortFatIsEquivalentToRef)
{
    set_up_apply_data();

    auto diag = y->extract_diagonal();
    auto ddiag = dy->extract_diagonal();

    GKO_ASSERT_MTX_NEAR(diag.get(), ddiag.get(), 0);
}


TEST_F(Dense, ExtractDiagonalOnShortFatIntoDenseCrossExecutor)
{
    set_up_apply_data();
    auto diag = Diagonal::create(ref, y->get_size()[0]);
    // test make_temporary_clone
    auto ddiag = Diagonal::create(ref, y->get_size()[0]);

    y->extract_diagonal(diag.get());
    dy->extract_diagonal(ddiag.get());

    GKO_ASSERT_MTX_NEAR(diag.get(), ddiag.get(), 0);
}


TEST_F(Dense, InplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    x->compute_absolute_inplace();
    dx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(x, dx, r<vtype>::value);
}


TEST_F(Dense, OutplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    auto abs_x = x->compute_absolute();
    auto dabs_x = dx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_x, dabs_x, r<vtype>::value);
}


TEST_F(Dense, OutplaceAbsoluteMatrixIntoDenseCrossExecutor)
{
    set_up_apply_data();
    auto abs_x = NormVector::create(ref, x->get_size());
    // test make_temporary_clone and non-default stride
    auto dabs_x = NormVector::create(ref, x->get_size(), x->get_size()[1] + 2);

    x->compute_absolute(abs_x.get());
    dx->compute_absolute(dabs_x.get());

    GKO_ASSERT_MTX_NEAR(abs_x, dabs_x, r<vtype>::value);
}


TEST_F(Dense, MakeComplexIsEquivalentToRef)
{
    set_up_apply_data();

    auto complex_x = x->make_complex();
    auto dcomplex_x = dx->make_complex();

    GKO_ASSERT_MTX_NEAR(complex_x, dcomplex_x, 0);
}


TEST_F(Dense, MakeComplexIntoDenseCrossExecutor)
{
    set_up_apply_data();
    auto complex_x = ComplexMtx::create(ref, x->get_size());
    // test make_temporary_clone and non-default stride
    auto dcomplex_x =
        ComplexMtx::create(ref, x->get_size(), x->get_size()[1] + 2);

    x->make_complex(complex_x.get());
    dx->make_complex(dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(complex_x, dcomplex_x, 0);
}


TEST_F(Dense, GetRealIsEquivalentToRef)
{
    set_up_apply_data();

    auto real_x = x->get_real();
    auto dreal_x = dx->get_real();

    GKO_ASSERT_MTX_NEAR(real_x, dreal_x, 0);
}


TEST_F(Dense, GetRealIntoDenseCrossExecutor)
{
    set_up_apply_data();
    auto real_x = Mtx::create(ref, x->get_size());
    // test make_temporary_clone and non-default stride
    auto dreal_x = Mtx::create(ref, x->get_size(), x->get_size()[1] + 2);

    x->get_real(real_x.get());
    dx->get_real(dreal_x.get());

    GKO_ASSERT_MTX_NEAR(real_x, dreal_x, 0);
}


TEST_F(Dense, GetImagIsEquivalentToRef)
{
    set_up_apply_data();

    auto imag_x = x->get_imag();
    auto dimag_x = dx->get_imag();

    GKO_ASSERT_MTX_NEAR(imag_x, dimag_x, 0);
}


TEST_F(Dense, GetImagIntoDenseCrossExecutor)
{
    set_up_apply_data();
    auto imag_x = Mtx::create(ref, x->get_size());
    // test make_temporary_clone and non-default stride
    auto dimag_x = Mtx::create(ref, x->get_size(), x->get_size()[1] + 2);

    x->get_imag(imag_x.get());
    dx->get_imag(dimag_x.get());

    GKO_ASSERT_MTX_NEAR(imag_x, dimag_x, 0);
}


}  // namespace
