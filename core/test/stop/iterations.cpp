/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <core/stop/iterations.hpp>


#include <gtest/gtest.h>


namespace {


constexpr unsigned int test_iterations = 10;


class Iterations : public ::testing::Test {
protected:
    Iterations()
        : factory_{gko::stop::Iterations::Factory::create(test_iterations)},
          exec_{gko::ReferenceExecutor::create()}
    {}

    std::unique_ptr<gko::stop::Iterations::Factory> factory_;
    std::shared_ptr<const gko::Executor> exec_;
};


TEST_F(Iterations, CanCreateFactory)
{
    ASSERT_NE(factory_, nullptr);
    ASSERT_EQ(factory_->v_, test_iterations);
}


TEST_F(Iterations, CanCreateCriterion)
{
    auto criterion = factory_->create_criterion(nullptr, nullptr, nullptr);
    ASSERT_NE(criterion, nullptr);
}


TEST_F(Iterations, WaitsTillIteration)
{
    auto criterion = factory_->create_criterion(nullptr, nullptr, nullptr);
    gko::Array<bool> converged(exec_, 1);

    ASSERT_FALSE(criterion->update()
                     .num_iterations(test_iterations - 1)
                     .check(converged));
    ASSERT_TRUE(
        criterion->update().num_iterations(test_iterations).check(converged));
    ASSERT_TRUE(criterion->update()
                    .num_iterations(test_iterations + 1)
                    .check(converged));
}


}  // namespace
