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

#include "core/factorization/par_ilut_kernels.hpp"


#include <algorithm>
#include <fstream>
#include <memory>
#include <random>
#include <string>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"
#include "matrices/config.hpp"


namespace {


class ParIlut : public ::testing::Test {
protected:
    using value_type = gko::default_precision;
    using index_type = gko::int32;
    using Dense = gko::matrix::Dense<value_type>;
    using ComplexDense = gko::matrix::Dense<std::complex<value_type>>;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using ComplexCsr = gko::matrix::Csr<std::complex<value_type>, index_type>;

    ParIlut()
        : mtx_size(700, 500),
          rand_engine(1337),
          ref(gko::ReferenceExecutor::create()),
          hip(gko::HipExecutor::create(0, ref))
    {
        mtx1 = gko::test::generate_random_matrix<Csr>(
            mtx_size[0], mtx_size[1],
            std::uniform_int_distribution<>(10, mtx_size[1]),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
        mtx2 = gko::test::generate_random_matrix<Csr>(
            mtx_size[0], mtx_size[1],
            std::uniform_int_distribution<>(0, mtx_size[1]),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
        mtx_l = gko::test::generate_random_lower_triangular_matrix<Csr>(
            mtx_size[0], mtx_size[0], false,
            std::uniform_int_distribution<>(10, mtx_size[0]),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
        mtx_l_complex =
            gko::test::generate_random_lower_triangular_matrix<ComplexCsr>(
                mtx_size[0], mtx_size[0], false,
                std::uniform_int_distribution<>(10, mtx_size[0]),
                std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
        mtx_u = gko::test::generate_random_upper_triangular_matrix<Csr>(
            mtx_size[0], mtx_size[0], false,
            std::uniform_int_distribution<>(10, mtx_size[0]),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
        mtx_u_complex =
            gko::test::generate_random_upper_triangular_matrix<ComplexCsr>(
                mtx_size[0], mtx_size[0], false,
                std::uniform_int_distribution<>(10, mtx_size[0]),
                std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
        alpha = gko::initialize<Dense>({1.0}, ref);
        beta = gko::initialize<Dense>({-2.0}, ref);
        alpha_complex = gko::initialize<ComplexDense>(
            {std::complex<value_type>{0., -1.}}, ref);
        beta_complex = gko::initialize<ComplexDense>(
            {std::complex<value_type>{-2., 1.}}, ref);

        dmtx1 = Csr::create(hip);
        dmtx1->copy_from(mtx1.get());
        dmtx2 = Csr::create(hip);
        dmtx2->copy_from(mtx2.get());
        dmtx_l = Csr::create(hip);
        dmtx_l->copy_from(mtx_l.get());
        dmtx_u = Csr::create(hip);
        dmtx_u->copy_from(mtx_u.get());
        dmtx_l_complex = ComplexCsr::create(hip);
        dmtx_l_complex->copy_from(mtx_l_complex.get());
        dmtx_u_complex = ComplexCsr::create(hip);
        dmtx_u_complex->copy_from(mtx_u_complex.get());
        dalpha = Dense::create(hip);
        dalpha->copy_from(alpha.get());
        dbeta = Dense::create(hip);
        dbeta->copy_from(beta.get());
        dalpha_complex = ComplexDense::create(hip);
        dalpha_complex->copy_from(alpha_complex.get());
        dbeta_complex = ComplexDense::create(hip);
        dbeta_complex->copy_from(beta_complex.get());
    }

    template <typename Mtx>
    void test_select(const std::unique_ptr<Mtx> &mtx,
                     const std::unique_ptr<Mtx> &dmtx, index_type rank)
    {
        auto size = index_type(mtx->get_num_stored_elements());

        auto res =
            gko::kernels::reference::par_ilut_factorization::threshold_select(
                ref, mtx->get_const_values(), size, rank);
        auto dres = gko::kernels::hip::par_ilut_factorization::threshold_select(
            hip, dmtx->get_const_values(), size, rank);

        if (gko::is_complex_s<typename Mtx::value_type>::value) {
            ASSERT_NEAR(res, dres, 1e-14);
        } else {
            ASSERT_EQ(res, dres);
        }
    }

    template <typename Mtx>
    void test_filter(const std::unique_ptr<Mtx> &mtx,
                     const std::unique_ptr<Mtx> &dmtx, value_type threshold,
                     bool lower)
    {
        gko::Array<index_type> new_row_ptrs(ref);
        gko::Array<index_type> new_col_idxs(ref);
        gko::Array<typename Mtx::value_type> new_vals(ref);
        gko::Array<index_type> dnew_row_ptrs(hip);
        gko::Array<index_type> dnew_col_idxs(hip);
        gko::Array<typename Mtx::value_type> dnew_vals(hip);

        gko::kernels::reference::par_ilut_factorization::threshold_filter(
            ref, mtx.get(), threshold, new_row_ptrs, new_col_idxs, new_vals,
            lower);
        gko::kernels::hip::par_ilut_factorization::threshold_filter(
            hip, dmtx.get(), threshold, dnew_row_ptrs, dnew_col_idxs, dnew_vals,
            lower);
        auto res =
            Mtx::create(ref, mtx_size, new_vals, new_col_idxs, new_row_ptrs);
        auto dres =
            Mtx::create(hip, mtx_size, dnew_vals, dnew_col_idxs, dnew_row_ptrs);

        GKO_ASSERT_MTX_NEAR(res, dres, 0);
        GKO_ASSERT_MTX_EQ_SPARSITY(res, dres);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::HipExecutor> hip;

    const gko::dim<2> mtx_size;
    std::default_random_engine rand_engine;

    std::unique_ptr<Csr> mtx1;
    std::unique_ptr<Csr> mtx2;
    std::unique_ptr<Csr> mtx_l;
    std::unique_ptr<ComplexCsr> mtx_l_complex;
    std::unique_ptr<Csr> mtx_u;
    std::unique_ptr<ComplexCsr> mtx_u_complex;
    std::unique_ptr<Dense> alpha;
    std::unique_ptr<Dense> beta;
    std::unique_ptr<ComplexDense> alpha_complex;
    std::unique_ptr<ComplexDense> beta_complex;

    std::unique_ptr<Csr> dmtx1;
    std::unique_ptr<Csr> dmtx2;
    std::unique_ptr<Csr> dmtx_l;
    std::unique_ptr<ComplexCsr> dmtx_l_complex;
    std::unique_ptr<Csr> dmtx_u;
    std::unique_ptr<ComplexCsr> dmtx_u_complex;
    std::unique_ptr<Dense> dalpha;
    std::unique_ptr<Dense> dbeta;
    std::unique_ptr<ComplexDense> dalpha_complex;
    std::unique_ptr<ComplexDense> dbeta_complex;
};


TEST_F(ParIlut, KernelThresholdSelectIsEquivalentToRef)
{
    test_select(mtx_l, dmtx_l, mtx_l->get_num_stored_elements() / 3);
}


TEST_F(ParIlut, KernelThresholdSelectMinIsEquivalentToRef)
{
    test_select(mtx_l, dmtx_l, 0);
}


TEST_F(ParIlut, KernelThresholdSelectMaxIsEquivalentToRef)
{
    test_select(mtx_l, dmtx_l, mtx_l->get_num_stored_elements() - 1);
}


TEST_F(ParIlut, KernelComplexThresholdSelectIsEquivalentToRef)
{
    test_select(mtx_l_complex, dmtx_l_complex,
                mtx_l_complex->get_num_stored_elements() / 3);
}


TEST_F(ParIlut, KernelComplexThresholdSelectMinIsEquivalentToRef)
{
    test_select(mtx_l_complex, dmtx_l_complex, 0);
}


TEST_F(ParIlut, KernelComplexThresholdSelectMaxLowerIsEquivalentToRef)
{
    test_select(mtx_l_complex, dmtx_l_complex,
                mtx_l_complex->get_num_stored_elements() - 1);
}


TEST_F(ParIlut, KernelThresholdFilterLowerIsEquivalentToRef)
{
    test_filter(mtx_l, dmtx_l, 0.5, true);
}


TEST_F(ParIlut, KernelThresholdFilterUpperIsEquivalentToRef)
{
    test_filter(mtx_u, dmtx_u, 0.5, false);
}


TEST_F(ParIlut, KernelThresholdFilterNoneLowerIsEquivalentToRef)
{
    test_filter(mtx_l, dmtx_l, 0, true);
}


TEST_F(ParIlut, KernelThresholdFilterNoneUpperIsEquivalentToRef)
{
    test_filter(mtx_u, dmtx_u, 0, false);
}


TEST_F(ParIlut, KernelThresholdFilterAllLowerIsEquivalentToRef)
{
    test_filter(mtx_l, dmtx_l, 1e6, true);
}


TEST_F(ParIlut, KernelThresholdFilterAllUpperIsEquivalentToRef)
{
    test_filter(mtx_u, dmtx_u, 1e6, false);
}


TEST_F(ParIlut, KernelComplexThresholdFilterLowerIsEquivalentToRef)
{
    test_filter(mtx_l_complex, dmtx_l_complex, 0.5, true);
}


TEST_F(ParIlut, KernelComplexThresholdFilterUpperIsEquivalentToRef)
{
    test_filter(mtx_u_complex, dmtx_u_complex, 0.5, false);
}


TEST_F(ParIlut, KernelComplexThresholdFilterNoneLowerIsEquivalentToRef)
{
    test_filter(mtx_l_complex, dmtx_l_complex, 0, true);
}


TEST_F(ParIlut, KernelComplexThresholdFilterNoneUpperIsEquivalentToRef)
{
    test_filter(mtx_u_complex, dmtx_u_complex, 0, false);
}


TEST_F(ParIlut, KernelComplexThresholdFilterAllLowerIsEquivalentToRef)
{
    test_filter(mtx_l_complex, dmtx_l_complex, 1e6, true);
}


TEST_F(ParIlut, KernelComplexThresholdFilterAllUpperIsEquivalentToRef)
{
    test_filter(mtx_u_complex, dmtx_u_complex, 1e6, false);
}


}  // namespace
