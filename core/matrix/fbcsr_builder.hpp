/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#ifndef GKO_CORE_MATRIX_FBCSR_BUILDER_HPP_
#define GKO_CORE_MATRIX_FBCSR_BUILDER_HPP_


#include <ginkgo/core/matrix/fbcsr.hpp>


namespace gko {
namespace matrix {


/**
 * @internal
 *
 * Allows intrusive access to the arrays stored within a @ref Fbcsr matrix.
 *
 * @tparam ValueType  the value type of the matrix
 * @tparam IndexType  the index type of the matrix
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class FbcsrBuilder {
public:
    /**
     * @return The column index array of the matrix.
     */
    array<IndexType>& get_col_idx_array() { return matrix_->col_idxs_; }

    /**
     * @return The value array of the matrix.
     */
    array<ValueType>& get_value_array() { return matrix_->values_; }

    /**
     * @return The (uniform) block size
     */
    int get_block_size() const { return matrix_->bs_; }

    /**
     * @param matrix  An existing FBCSR matrix
     *                for which intrusive access is needed
     */
    explicit FbcsrBuilder(
        pointer_param<Fbcsr<ValueType, IndexType>> const matrix)
        : matrix_{matrix.get()}
    {}

    ~FbcsrBuilder() = default;

    // make this type non-movable
    FbcsrBuilder(const FbcsrBuilder&) = delete;
    FbcsrBuilder(FbcsrBuilder&&) = delete;
    FbcsrBuilder& operator=(const FbcsrBuilder&) = delete;
    FbcsrBuilder& operator=(FbcsrBuilder&&) = delete;

private:
    Fbcsr<ValueType, IndexType>* matrix_;
};


}  // namespace matrix
}  // namespace gko

#endif  // GKO_CORE_MATRIX_FBCSR_BUILDER_HPP_
