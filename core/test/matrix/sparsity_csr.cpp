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

#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>


namespace {


class SparsityCsr : public ::testing::Test {
protected:
    using Mtx = gko::matrix::SparsityCsr<>;

    SparsityCsr()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::matrix::SparsityCsr<>::create(exec, gko::dim<2>{2, 3}, 4))
    {
        Mtx::index_type *c = mtx->get_col_idxs();
        Mtx::index_type *r = mtx->get_row_ptrs();
        r[0] = 0;
        r[1] = 3;
        r[2] = 4;
        c[0] = 0;
        c[1] = 1;
        c[2] = 2;
        c[3] = 1;
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;

    void assert_equal_to_original_mtx(const Mtx *m)
    {
        auto c = m->get_const_col_idxs();
        auto r = m->get_const_row_ptrs();
        auto v = m->get_const_value();
        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_nonzeros(), 4);
        EXPECT_EQ(r[0], 0);
        EXPECT_EQ(r[1], 3);
        EXPECT_EQ(r[2], 4);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[2], 2);
        EXPECT_EQ(c[3], 1);
        EXPECT_EQ(v[0], 1.0);
    }

    void assert_empty(Mtx *m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_num_nonzeros(), 0);
        ASSERT_EQ(m->get_const_col_idxs(), nullptr);
        ASSERT_EQ(m->get_const_row_ptrs(), nullptr);
        ASSERT_EQ(m->get_const_value(), nullptr);
        ASSERT_EQ(m->get_col_idxs(), nullptr);
        ASSERT_EQ(m->get_row_ptrs(), nullptr);
        ASSERT_EQ(m->get_value(), nullptr);
    }
};


TEST_F(SparsityCsr, KnowsItsSize)
{
    ASSERT_EQ(mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(mtx->get_num_nonzeros(), 4);
}


TEST_F(SparsityCsr, ContainsCorrectData)
{
    assert_equal_to_original_mtx(mtx.get());
}


TEST_F(SparsityCsr, CanBeEmpty)
{
    auto mtx = Mtx::create(exec);

    assert_empty(mtx.get());
}


TEST_F(SparsityCsr, SetsCorrectDefaultValue)
{
    auto mtx = gko::matrix::SparsityCsr<>::create(
        exec, gko::dim<2>{3, 2}, static_cast<gko::size_type>(0));

    ASSERT_EQ(mtx->get_const_value()[0], 1.0);
    ASSERT_EQ(mtx->get_value()[0], 1.0);
}


TEST_F(SparsityCsr, CanBeCreatedFromExistingData)
{
    gko::int32 col_idxs[] = {0, 1, 1, 0};
    gko::int32 row_ptrs[] = {0, 2, 3, 4};

    auto mtx = gko::matrix::SparsityCsr<>::create(
        exec, gko::dim<2>{3, 2},
        gko::Array<gko::int32>::view(exec, 4, col_idxs),
        gko::Array<gko::int32>::view(exec, 4, row_ptrs), 2.0);

    ASSERT_EQ(mtx->get_const_col_idxs(), col_idxs);
    ASSERT_EQ(mtx->get_const_row_ptrs(), row_ptrs);
    ASSERT_EQ(mtx->get_const_value()[0], 2.0);
    ASSERT_EQ(mtx->get_col_idxs(), col_idxs);
    ASSERT_EQ(mtx->get_row_ptrs(), row_ptrs);
    ASSERT_EQ(mtx->get_value()[0], 2.0);
}


TEST_F(SparsityCsr, CanBeCopied)
{
    auto copy = Mtx::create(exec);

    copy->copy_from(mtx.get());

    assert_equal_to_original_mtx(mtx.get());
    assert_equal_to_original_mtx(copy.get());
}


TEST_F(SparsityCsr, CanBeMoved)
{
    auto copy = Mtx::create(exec);

    copy->copy_from(std::move(mtx));

    assert_equal_to_original_mtx(copy.get());
}


TEST_F(SparsityCsr, CanBeCloned)
{
    auto clone = mtx->clone();

    assert_equal_to_original_mtx(mtx.get());
    assert_equal_to_original_mtx(dynamic_cast<Mtx *>(clone.get()));
}


TEST_F(SparsityCsr, CanBeCleared)
{
    mtx->clear();

    assert_empty(mtx.get());
}


TEST_F(SparsityCsr, CanBeReadFromMatrixData)
{
    auto m = Mtx::create(exec);

    m->read({{2, 3},
             {{0, 0, 1.0},
              {0, 1, 3.0},
              {0, 2, 2.0},
              {1, 0, 0.0},
              {1, 1, 5.0},
              {1, 2, 0.0}}});

    assert_equal_to_original_mtx(m.get());
}


TEST_F(SparsityCsr, GeneratesCorrectMatrixData)
{
    using tpl = gko::matrix_data<>::nonzero_type;
    gko::matrix_data<> data;

    mtx->write(data);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    ASSERT_EQ(data.nonzeros.size(), 4);
    EXPECT_EQ(data.nonzeros[0], tpl(0, 0, 1.0));
    EXPECT_EQ(data.nonzeros[1], tpl(0, 1, 1.0));
    EXPECT_EQ(data.nonzeros[2], tpl(0, 2, 1.0));
    EXPECT_EQ(data.nonzeros[3], tpl(1, 1, 1.0));
}


}  // namespace
