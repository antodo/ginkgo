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

// TODO remove when the HIP includes are fixed
#include <hip/hip_runtime.h>


#include "hip/components/searching.hip.hpp"


#include <algorithm>
#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>


namespace {


using namespace gko::kernels::hip;


class Merging : public ::testing::Test {
protected:
    Merging()
        : ref(gko::ReferenceExecutor::create()),
          hip(gko::HipExecutor::create(0, ref)),
          rng(123456),
          max_size{1637},
          sizes{},
          data1(ref, max_size),
          data2(ref, max_size),
          outdata(ref, 2 * max_size),
          refdata(ref, 2 * max_size),
          ddata1(hip),
          ddata2(hip),
          doutdata(hip, 2 * max_size)
    {}

    void init_data()
    {
        std::uniform_int_distribution<gko::int32> dist(0, max_size);
        for (auto i = 0; i < max_size; ++i) {
            data1.get_data()[i] = dist(rng);
            data2.get_data()[i] = dist(rng);
        }
        std::sort(data1.get_data(), data1.get_data() + max_size);
        std::sort(data2.get_data(), data2.get_data() + max_size);

        ddata1 = data1;
        ddata2 = data2;
    }

    void assert_eq_ref(int size, int eq_size)
    {
        outdata = doutdata;
        auto out_ptr = outdata.get_const_data();
        auto out_end = out_ptr + eq_size;
        auto ref_ptr = refdata.get_data();
        std::copy_n(data1.get_const_data(), size, ref_ptr);
        std::copy_n(data2.get_const_data(), size, ref_ptr + size);
        std::sort(ref_ptr, ref_ptr + 2 * size);

        ASSERT_TRUE(std::equal(out_ptr, out_end, ref_ptr));
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::HipExecutor> hip;
    std::default_random_engine rng;

    int max_size;
    std::vector<int> sizes;
    gko::Array<gko::int32> data1;
    gko::Array<gko::int32> data2;
    gko::Array<gko::int32> outdata;
    gko::Array<gko::int32> refdata;
    gko::Array<gko::int32> ddata1;
    gko::Array<gko::int32> ddata2;
    gko::Array<gko::int32> doutdata;
};


__global__ test_merge_step(const gko::int32 *a, const gko::int32 *b,
                           gko::int32 *c)
{
    auto warp = tiled_partition<config::warp_size>(this_thread_block());
    auto i = warp.thread_rank();
    auto result = kernel::group_merge_step(a[i], b[i], config::warp_size, warp);
    c[i] = min(result.a_val, result.b_val);
}

TEST_F(Merging, MergeStep)
{
    for (auto i = 0; i < rng_runs; ++i) {
        init_data();
        test_merge_step<<<1, config::warp_size>>>(ddata1.get_const_data(),
                                                  ddata2.get_const_data(),
                                                  doutdata.get_data());

        assert_eq_ref(config::warp_size, config::warp_size);
    }
}


__global__ test_merge(const gko::int32 *a, const gko::int32 *b, int size,
                      gko::int32 *c)
{
    auto warp = tiled_partition<config::warp_size>(this_thread_block());
    kernel::group_merge(
        a, size, b, size, warp,
        [&](int a_idx, gko::int32 a_val, int b_idx, gko::int32 b_val, int i) {
            c[i] = min(a_val, b_val);
        });
}

TEST_F(Merging, FullMerge)
{
    for (auto i = 0; i < rng_runs; ++i) {
        init_data();
        for (auto size : sizes) {
            test_merge_step<<<1, config::warp_size>>>(
                ddata1.get_const_data(), ddata2.get_const_data(), size,
                doutdata.get_data());

            assert_eq_ref(size, 2 * size);
        }
    }
}


}  // namespace
