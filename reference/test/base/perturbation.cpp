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

#include <ginkgo/core/base/perturbation.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils/assertions.hpp"


namespace {


class Perturbation : public ::testing::Test {
protected:
    using mtx = gko::matrix::Dense<>;

    Perturbation()
        : exec{gko::ReferenceExecutor::create()},
          basis{gko::initialize<mtx>({2.0, 1.0}, exec)},
          projector{gko::initialize<mtx>({{3.0, 2.0}}, exec)},
          scalar{gko::initialize<mtx>({2.0}, exec)}
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<gko::LinOp> basis;
    std::shared_ptr<gko::LinOp> projector;
    std::shared_ptr<gko::LinOp> scalar;
};


TEST_F(Perturbation, AppliesToVector)
{
    /*
        cmp = I + 2 * [ 2 ] * [ 3 2 ]
                      [ 1 ]
    */
    auto cmp = gko::Perturbation<>::create(scalar, basis, projector);
    auto x = gko::initialize<mtx>({1.0, 2.0}, exec);
    auto res = mtx::create_with_config_of(gko::lend(x));

    cmp->apply(gko::lend(x), gko::lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({29.0, 16.0}), 1e-15);
}


TEST_F(Perturbation, AppliesLinearCombinationToVector)
{
    /*
        cmp = I + 2 * [ 2 ] * [ 3 2 ]
                      [ 1 ]
    */
    auto cmp = gko::Perturbation<>::create(scalar, basis, projector);
    auto alpha = gko::initialize<mtx>({3.0}, exec);
    auto beta = gko::initialize<mtx>({-1.0}, exec);
    auto x = gko::initialize<mtx>({1.0, 2.0}, exec);
    auto res = gko::clone(x);

    cmp->apply(gko::lend(alpha), gko::lend(x), gko::lend(beta), gko::lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({86.0, 46.0}), 1e-15);
}


TEST_F(Perturbation, ConstructionByBasisAppliesToVector)
{
    /*
        cmp = I + 2 * [ 2 ] * [ 2 1 ]
                      [ 1 ]
    */
    auto cmp = gko::Perturbation<>::create(scalar, basis);
    auto x = gko::initialize<mtx>({1.0, 2.0}, exec);
    auto res = mtx::create_with_config_of(gko::lend(x));

    cmp->apply(gko::lend(x), gko::lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({17.0, 10.0}), 1e-15);
}


TEST_F(Perturbation, ConstructionByBasisAppliesLinearCombinationToVector)
{
    /*
        cmp = I + 2 * [ 2 ] * [ 2 1 ]
                      [ 1 ]
    */
    auto cmp = gko::Perturbation<>::create(scalar, basis);
    auto alpha = gko::initialize<mtx>({3.0}, exec);
    auto beta = gko::initialize<mtx>({-1.0}, exec);
    auto x = gko::initialize<mtx>({1.0, 2.0}, exec);
    auto res = gko::clone(x);

    cmp->apply(gko::lend(alpha), gko::lend(x), gko::lend(beta), gko::lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({50.0, 28.0}), 1e-15);
}


}  // namespace
