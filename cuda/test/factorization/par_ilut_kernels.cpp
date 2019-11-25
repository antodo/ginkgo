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
        mtx2 = gko::test::generate_random_matrix<Csr>(
            mtx_size[0], mtx_size[1],
            std::uniform_int_distribution<>(0, mtx_size[1]),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
        alpha = gko::initialize<Dense>({1.0}, ref);
        beta = gko::initialize<Dense>({-2.0}, ref);

        dmtx1 = Csr::create(cuda);
        dmtx1->copy_from(mtx1.get());
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
    std::unique_ptr<Csr> mtx2;
    std::unique_ptr<Dense> alpha;
    std::unique_ptr<Dense> beta;

    std::unique_ptr<Csr> dmtx1;
    std::unique_ptr<Csr> dmtx2;
    std::unique_ptr<Dense> dalpha;
    std::unique_ptr<Dense> dbeta;
};


TEST_F(ParIlut, CudaKernelSpGeAMIsEquivalentToRef)
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
}

}  // namespace
