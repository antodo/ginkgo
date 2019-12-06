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
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using ComplexCsr = gko::matrix::Csr<std::complex<value_type>, index_type>;

    ParIlut()
        : mtx_size(532, 423),
          rand_engine(1337),
          ref(gko::ReferenceExecutor::create()),
          cuda(gko::CudaExecutor::create(0, ref))
    {
        mtx1 = gko::test::generate_random_matrix<Csr>(
            mtx_size[0], mtx_size[1],
            std::uniform_int_distribution<>(10, mtx_size[1]),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
        mtx1_complex = gko::test::generate_random_matrix<ComplexCsr>(
            mtx_size[0], mtx_size[1],
            std::uniform_int_distribution<>(10, mtx_size[1]),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
        mtx2 = gko::test::generate_random_matrix<Csr>(
            mtx_size[0], mtx_size[1],
            std::uniform_int_distribution<>(0, mtx_size[1]),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
        alpha = gko::initialize<Dense>({1.0}, ref);
        beta = gko::initialize<Dense>({-2.0}, ref);

        dmtx1 = Csr::create(cuda);
        dmtx1->copy_from(mtx1.get());
        dmtx1_complex = ComplexCsr::create(cuda);
        dmtx1_complex->copy_from(mtx1_complex.get());
        dmtx2 = Csr::create(cuda);
        dmtx2->copy_from(mtx2.get());
        dalpha = Dense::create(cuda);
        dalpha->copy_from(alpha.get());
        dbeta = Dense::create(cuda);
        dbeta->copy_from(beta.get());
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::CudaExecutor> cuda;

    const gko::dim<2> mtx_size;
    std::default_random_engine rand_engine;

    std::unique_ptr<Csr> mtx1;
    std::unique_ptr<ComplexCsr> mtx1_complex;
    std::unique_ptr<Csr> mtx2;
    std::unique_ptr<Dense> alpha;
    std::unique_ptr<Dense> beta;

    std::unique_ptr<Csr> dmtx1;
    std::unique_ptr<ComplexCsr> dmtx1_complex;
    std::unique_ptr<Csr> dmtx2;
    std::unique_ptr<Dense> dalpha;
    std::unique_ptr<Dense> dbeta;
};


TEST_F(ParIlut, KernelThresholdSelectIsEquivalentToRef)
{
    gko::Array<index_type> new_row_ptrs(ref);
    auto size = index_type(mtx1->get_num_stored_elements());
    auto rank = size / 3;

    auto res =
        gko::kernels::reference::par_ilut_factorization::threshold_select(
            ref, mtx1->get_const_values(), size, rank);
    auto dres = gko::kernels::cuda::par_ilut_factorization::threshold_select(
        cuda, dmtx1->get_const_values(), size, rank);

    ASSERT_EQ(res, dres);
}


TEST_F(ParIlut, KernelThresholdSelectMinIsEquivalentToRef)
{
    gko::Array<index_type> new_row_ptrs(ref);
    auto size = index_type(mtx1->get_num_stored_elements());
    auto rank = index_type(0);

    auto res =
        gko::kernels::reference::par_ilut_factorization::threshold_select(
            ref, mtx1->get_const_values(), size, rank);
    auto dres = gko::kernels::cuda::par_ilut_factorization::threshold_select(
        cuda, dmtx1->get_const_values(), size, rank);

    ASSERT_EQ(res, dres);
}


TEST_F(ParIlut, KernelThresholdSelectMaxIsEquivalentToRef)
{
    gko::Array<index_type> new_row_ptrs(ref);
    auto size = index_type(mtx1->get_num_stored_elements());
    auto rank = index_type(size - 1);

    auto res =
        gko::kernels::reference::par_ilut_factorization::threshold_select(
            ref, mtx1->get_const_values(), size, rank);
    auto dres = gko::kernels::cuda::par_ilut_factorization::threshold_select(
        cuda, dmtx1->get_const_values(), size, rank);

    ASSERT_EQ(res, dres);
}


TEST_F(ParIlut, KernelComplexThresholdSelectIsEquivalentToRef)
{
    gko::Array<index_type> new_row_ptrs(ref);
    auto size = index_type(mtx1->get_num_stored_elements());
    auto rank = size / 3;

    auto res =
        gko::kernels::reference::par_ilut_factorization::threshold_select<
            std::complex<value_type>>(ref, mtx1_complex->get_const_values(),
                                      size, rank);
    auto dres = gko::kernels::cuda::par_ilut_factorization::threshold_select<
        std::complex<value_type>>(cuda, dmtx1_complex->get_const_values(), size,
                                  rank);

    // host and device code might calculate abs(complex) differently
    ASSERT_NEAR(res, dres, 1e-14);
}


TEST_F(ParIlut, KernelComplexThresholdSelectMinIsEquivalentToRef)
{
    gko::Array<index_type> new_row_ptrs(ref);
    auto size = index_type(mtx1->get_num_stored_elements());
    auto rank = index_type(0);

    auto res =
        gko::kernels::reference::par_ilut_factorization::threshold_select<
            std::complex<value_type>>(ref, mtx1_complex->get_const_values(),
                                      size, rank);
    auto dres = gko::kernels::cuda::par_ilut_factorization::threshold_select<
        std::complex<value_type>>(cuda, dmtx1_complex->get_const_values(), size,
                                  rank);

    // host and device code might calculate abs(complex) differently
    ASSERT_NEAR(res, dres, 1e-14);
}


TEST_F(ParIlut, KernelComplexThresholdSelectMaxIsEquivalentToRef)
{
    gko::Array<index_type> new_row_ptrs(ref);
    auto size = index_type(mtx1->get_num_stored_elements());
    auto rank = index_type(size - 1);

    auto res =
        gko::kernels::reference::par_ilut_factorization::threshold_select<
            std::complex<value_type>>(ref, mtx1_complex->get_const_values(),
                                      size, rank);
    auto dres = gko::kernels::cuda::par_ilut_factorization::threshold_select<
        std::complex<value_type>>(cuda, dmtx1_complex->get_const_values(), size,
                                  rank);

    // host and device code might calculate abs(complex) differently
    ASSERT_NEAR(res, dres, 1e-14);
}


TEST_F(ParIlut, KernelThresholdFilterIsEquivalentToRef)
{
    gko::Array<index_type> new_row_ptrs(ref);
    gko::Array<index_type> new_col_idxs(ref);
    gko::Array<value_type> new_vals(ref);
    gko::Array<index_type> dnew_row_ptrs(cuda);
    gko::Array<index_type> dnew_col_idxs(cuda);
    gko::Array<value_type> dnew_vals(cuda);
    value_type threshold{0.5};

    gko::kernels::reference::par_ilut_factorization::threshold_filter(
        ref, mtx1.get(), threshold, new_row_ptrs, new_col_idxs, new_vals);
    gko::kernels::cuda::par_ilut_factorization::threshold_filter(
        cuda, dmtx1.get(), threshold, dnew_row_ptrs, dnew_col_idxs, dnew_vals);
    auto res = Csr::create(ref, mtx_size, new_vals, new_col_idxs, new_row_ptrs);
    auto dres =
        Csr::create(cuda, mtx_size, dnew_vals, dnew_col_idxs, dnew_row_ptrs);

    GKO_ASSERT_MTX_NEAR(res, dres, 0);
    GKO_ASSERT_MTX_EQ_SPARSITY(res, dres);
}


TEST_F(ParIlut, KernelThresholdFilterNoneIsEquivalentToRef)
{
    gko::Array<index_type> new_row_ptrs(ref);
    gko::Array<index_type> new_col_idxs(ref);
    gko::Array<value_type> new_vals(ref);
    gko::Array<index_type> dnew_row_ptrs(cuda);
    gko::Array<index_type> dnew_col_idxs(cuda);
    gko::Array<value_type> dnew_vals(cuda);
    value_type threshold = 0;

    gko::kernels::reference::par_ilut_factorization::threshold_filter(
        ref, mtx1.get(), threshold, new_row_ptrs, new_col_idxs, new_vals);
    gko::kernels::cuda::par_ilut_factorization::threshold_filter(
        cuda, dmtx1.get(), threshold, dnew_row_ptrs, dnew_col_idxs, dnew_vals);
    auto res = Csr::create(ref, mtx_size, new_vals, new_col_idxs, new_row_ptrs);
    auto dres =
        Csr::create(cuda, mtx_size, dnew_vals, dnew_col_idxs, dnew_row_ptrs);

    GKO_ASSERT_MTX_NEAR(res, dres, 0);
    GKO_ASSERT_MTX_EQ_SPARSITY(res, dres);
}


TEST_F(ParIlut, KernelComplexThresholdFilterIsEquivalentToRef)
{
    gko::Array<index_type> new_row_ptrs(ref);
    gko::Array<index_type> new_col_idxs(ref);
    gko::Array<std::complex<value_type>> new_vals(ref);
    gko::Array<index_type> dnew_row_ptrs(cuda);
    gko::Array<index_type> dnew_col_idxs(cuda);
    gko::Array<std::complex<value_type>> dnew_vals(cuda);
    value_type threshold{0.5};

    gko::kernels::reference::par_ilut_factorization::threshold_filter<
        std::complex<value_type>>(ref, mtx1_complex.get(), threshold,
                                  new_row_ptrs, new_col_idxs, new_vals);
    gko::kernels::cuda::par_ilut_factorization::threshold_filter<
        std::complex<value_type>>(cuda, dmtx1_complex.get(), threshold,
                                  dnew_row_ptrs, dnew_col_idxs, dnew_vals);
    auto res =
        ComplexCsr::create(ref, mtx_size, new_vals, new_col_idxs, new_row_ptrs);
    auto dres = ComplexCsr::create(cuda, mtx_size, dnew_vals, dnew_col_idxs,
                                   dnew_row_ptrs);

    GKO_ASSERT_MTX_NEAR(res, dres, 0);
    GKO_ASSERT_MTX_EQ_SPARSITY(res, dres);
}


TEST_F(ParIlut, KernelComplexThresholdFilterNoneIsEquivalentToRef)
{
    gko::Array<index_type> new_row_ptrs(ref);
    gko::Array<index_type> new_col_idxs(ref);
    gko::Array<std::complex<value_type>> new_vals(ref);
    gko::Array<index_type> dnew_row_ptrs(cuda);
    gko::Array<index_type> dnew_col_idxs(cuda);
    gko::Array<std::complex<value_type>> dnew_vals(cuda);
    value_type threshold = 0;

    gko::kernels::reference::par_ilut_factorization::threshold_filter<
        std::complex<value_type>>(ref, mtx1_complex.get(), threshold,
                                  new_row_ptrs, new_col_idxs, new_vals);
    gko::kernels::cuda::par_ilut_factorization::threshold_filter<
        std::complex<value_type>>(cuda, dmtx1_complex.get(), threshold,
                                  dnew_row_ptrs, dnew_col_idxs, dnew_vals);
    auto res =
        ComplexCsr::create(ref, mtx_size, new_vals, new_col_idxs, new_row_ptrs);
    auto dres = ComplexCsr::create(cuda, mtx_size, dnew_vals, dnew_col_idxs,
                                   dnew_row_ptrs);

    GKO_ASSERT_MTX_NEAR(res, dres, 0);
    GKO_ASSERT_MTX_EQ_SPARSITY(res, dres);
}


TEST_F(ParIlut, KernelSpGeAMIsEquivalentToRef)
{
    gko::Array<index_type> new_row_ptrs(ref);
    gko::Array<index_type> new_col_idxs(ref);
    gko::Array<value_type> new_vals(ref);
    gko::Array<index_type> dnew_row_ptrs(cuda);
    gko::Array<index_type> dnew_col_idxs(cuda);
    gko::Array<value_type> dnew_vals(cuda);

    gko::kernels::reference::par_ilut_factorization::spgeam(
        ref, alpha.get(), mtx1.get(), beta.get(), mtx2.get(), new_row_ptrs,
        new_col_idxs, new_vals);
    gko::kernels::cuda::par_ilut_factorization::spgeam(
        cuda, dalpha.get(), dmtx1.get(), dbeta.get(), dmtx2.get(),
        dnew_row_ptrs, dnew_col_idxs, dnew_vals);
    auto res = Csr::create(ref, mtx_size, new_vals, new_col_idxs, new_row_ptrs);
    auto dres =
        Csr::create(cuda, mtx_size, dnew_vals, dnew_col_idxs, dnew_row_ptrs);

    GKO_ASSERT_MTX_NEAR(res, dres, 1e-14);
    GKO_ASSERT_MTX_EQ_SPARSITY(res, dres);
}

}  // namespace
