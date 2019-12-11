/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include <ginkgo/core/factorization/par_ilut.hpp>


#include <algorithm>
#include <memory>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/factorization/par_ilut_kernels.hpp"
#include "core/test/utils/assertions.hpp"


namespace {


class ParIlut : public ::testing::Test {
protected:
    using value_type = gko::default_precision;
    using index_type = gko::int32;
    using Dense = gko::matrix::Dense<value_type>;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using ComplexCsr = gko::matrix::Csr<std::complex<value_type>, index_type>;

    ParIlut()
        : i{0., 1.},
          ref(gko::ReferenceExecutor::create()),
          exec(std::static_pointer_cast<const gko::Executor>(ref)),

          mtx1(gko::initialize<Csr>({{.1, .1, -1., -2.},
                                     {0., .1, -2., -3.},
                                     {0., 0., -1., -1.},
                                     {0., 0., 0., 1.}},
                                    ref)),
          mtx1_expect_thrm2(gko::initialize<Csr>({{.1, 0., 0., -2.},
                                                  {0., .1, -2., -3.},
                                                  {0., 0., -1., 0.},
                                                  {0., 0., 0., 1.}},
                                                 ref)),
          mtx1_expect_thrm3(gko::initialize<Csr>({{.1, 0., 0., 0.},
                                                  {0., .1, 0., -3.},
                                                  {0., 0., -1., 0.},
                                                  {0., 0., 0., 1.}},
                                                 ref)),
          mtx1_complex(gko::initialize<ComplexCsr>(
              {{.1 + 0. * i, -1. + .1 * i, -1. + i, 1. - 2. * i},
               {0. * i, .1 - i, -2. + .2 * i, -3. - .1 * i},
               {0. * i, 0. * i, -1. - .3 * i, -1. + .1 * i},
               {0. * i, 0. * i, 0. * i, .1 + 2. * i}},
              ref)),
          mtx1_expect_complex_thrm(gko::initialize<ComplexCsr>(
              {{.1 + 0. * i, 0. * i, -1. + i, 1. - 2. * i},
               {0. * i, .1 - i, -2. + .2 * i, -3. - .1 * i},
               {0. * i, 0. * i, -1. - .3 * i, 0. * i},
               {0. * i, 0. * i, 0. * i, .1 + 2. * i}},
              ref)),
          mtx2(gko::initialize<Csr>({{0., 0., -1., -2.},
                                     {0., 0., 0., -3.},
                                     {0., 1., 0., 0.},
                                     {1., -1., 0., 0.}},
                                    ref)),
          mtx3(gko::initialize<Csr>({{1., 0., -1., 0.},
                                     {0., 1., -2., 3.},
                                     {3., 2., -1., -1.},
                                     {0., 0., 0., 0.}},
                                    ref)),
          mtx23_expect_geam(gko::initialize<Csr>({{-2., 0., 1., -2.},
                                                  {0., -2., 4., -9.},
                                                  {-6., -3., 2., 2.},
                                                  {1., -1., 0., 0.}},
                                                 ref))
    {}

    template <typename Mtx>
    void test_select(const std::unique_ptr<Mtx> &mtx, index_type rank,
                     value_type expected, value_type tolerance = 0.0)
    {
        auto vals = mtx->get_const_values();
        auto size = index_type(mtx->get_num_stored_elements());

        auto result =
            gko::kernels::reference::par_ilut_factorization::threshold_select(
                ref, vals, size, rank);

        ASSERT_NEAR(result, expected, tolerance);
    }

    template <typename Mtx>
    void test_filter(const std::unique_ptr<Mtx> &mtx, value_type threshold,
                     const std::unique_ptr<Mtx> &expected)
    {
        gko::Array<index_type> new_row_ptrs(exec);
        gko::Array<index_type> new_col_idxs(exec);
        gko::Array<typename Mtx::value_type> new_vals(exec);

        gko::kernels::reference::par_ilut_factorization::threshold_filter(
            ref, mtx.get(), threshold, new_row_ptrs, new_col_idxs, new_vals,
            false);
        auto res_mtx = Mtx::create(exec, mtx->get_size(), new_vals,
                                   new_col_idxs, new_row_ptrs);

        GKO_ASSERT_MTX_NEAR(expected, res_mtx, 0);
        GKO_ASSERT_MTX_EQ_SPARSITY(expected, res_mtx);
    }

    std::complex<value_type> i;

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Csr> mtx1;
    std::unique_ptr<Csr> mtx1_expect_thrm2;
    std::unique_ptr<Csr> mtx1_expect_thrm3;
    std::unique_ptr<ComplexCsr> mtx1_complex;
    std::unique_ptr<ComplexCsr> mtx1_expect_complex_thrm;
    std::unique_ptr<Csr> mtx2;
    std::unique_ptr<Csr> mtx3;
    std::unique_ptr<Csr> mtx23_expect_geam;
};


TEST_F(ParIlut, KernelThresholdSelect) { test_select(mtx1, 7, 2.0); }


TEST_F(ParIlut, KernelThresholdSelectMin) { test_select(mtx1, 0, 0.1); }


TEST_F(ParIlut, KernelThresholdSelectMax) { test_select(mtx1, 9, 3.0); }


TEST_F(ParIlut, KernelComplexThresholdSelect)
{
    test_select(mtx1_complex, 5, sqrt(2), 1e-14);
}


TEST_F(ParIlut, KernelComplexThresholdSelectMin)
{
    test_select(mtx1_complex, 0, 0.1, 1e-14);
}


TEST_F(ParIlut, KernelComplexThresholdSelectMax)
{
    test_select(mtx1_complex, 9, sqrt(9.01), 1e-14);
}


TEST_F(ParIlut, KernelThresholdFilterNone) { test_filter(mtx1, 0.0, mtx1); }


TEST_F(ParIlut, KernelThresholdFilterSomeAtThreshold)
{
    test_filter(mtx1, 2.0, mtx1_expect_thrm2);
}


TEST_F(ParIlut, KernelThresholdFilterSomeAboveThreshold)
{
    test_filter(mtx1, 3.0, mtx1_expect_thrm3);
}


TEST_F(ParIlut, KernelComplexThresholdFilterNone)
{
    test_filter(mtx1_complex, 0.0, mtx1_complex);
}


TEST_F(ParIlut, KernelComplexThresholdFilterSomeAtThreshold)
{
    test_filter(mtx1_complex, 1.01, mtx1_expect_complex_thrm);
}


TEST_F(ParIlut, KernelSpGeAM)
{
    gko::Array<index_type> new_row_ptrs(exec);
    gko::Array<index_type> new_col_idxs(exec);
    gko::Array<value_type> new_vals(exec);
    auto alpha = gko::initialize<Dense>({1.0}, exec);
    auto beta = gko::initialize<Dense>({-2.0}, exec);

    gko::kernels::reference::par_ilut_factorization::spgeam(
        ref, alpha.get(), mtx2.get(), beta.get(), mtx3.get(), new_row_ptrs,
        new_col_idxs, new_vals);
    auto res_mtx = Csr::create(exec, mtx1->get_size(), new_vals, new_col_idxs,
                               new_row_ptrs);

    GKO_ASSERT_MTX_NEAR(mtx23_expect_geam, res_mtx, 0);
    GKO_ASSERT_MTX_EQ_SPARSITY(mtx23_expect_geam, res_mtx);
}


}  // namespace
