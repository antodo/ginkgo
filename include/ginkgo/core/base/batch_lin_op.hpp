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

#ifndef GKO_PUBLIC_CORE_BASE_BATCH_LIN_OP_HPP_
#define GKO_PUBLIC_CORE_BASE_BATCH_LIN_OP_HPP_


#include <memory>
#include <type_traits>
#include <utility>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/matrix_assembly_data.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/log/logger.hpp>


namespace gko {
namespace batch {


/**
 * @addtogroup BatchLinOp
 *
 * @section batch_linop_concept Batched Linear operator as a concept
 *
 * A batch linear operator (BatchLinOp) forms the base class for all batched
 * linear algebra objects. In general, it follows the same structure as the
 * LinOp class, but has some crucial differences which make it not strictly
 * representable through or with the LinOp class.
 *
 * A batched operator is defined as a set of independent linear operators which
 * have no communication/information exchange between them. Therefore, any
 * collective operations between the batches is not possible and not
 * implemented. This allows for each batch to be computed and operated on in an
 * embarrassingly parallel fashion.
 *
 * A key difference between the LinOp and the BatchLinOp class is that the apply
 * between BatchLinOps is no longer supported. The user can apply a BatchLinOp
 * to a batch::MultiVector but not to any general BatchLinOp. This apply to a
 * batch::MultiVector is handled by the concrete LinOp and may be moved to the
 * base BatchLinOp class in the future.
 *
 * Therefore, the BatchLinOp serves only as a base class providing necessary
 * core functionality from Polymorphic object and store the dimensions of the
 * batched object.
 *
 * @ref BatchLinOp
 */
class BatchLinOp : public EnableAbstractPolymorphicObject<BatchLinOp> {
public:
    /**
     * Returns the number of items in the batch operator.
     *
     * @return  number of items in the batch operator
     */
    size_type get_num_batch_items() const noexcept
    {
        return get_size().get_num_batch_items();
    }

    /**
     * Returns the common size of the batch items.
     *
     * @return  the common size stored
     */
    dim<2> get_common_size() const { return get_size().get_common_size(); }

    /**
     * Returns the size of the batch operator.
     *
     * @return  size of the batch operator, a batch_dim object
     */
    const batch_dim<2>& get_size() const noexcept { return size_; }

protected:
    /**
     * Sets the size of the batch operator.
     *
     * @param size to be set
     */
    void set_size(const batch_dim<2>& size) { size_ = size; }

    /**
     * Creates a batch operator storing items of uniform sizes.
     *
     * @param exec        the executor where all the operations are performed
     * @param batch_size  the size the batched operator, as a batch_dim object
     */
    explicit BatchLinOp(std::shared_ptr<const Executor> exec,
                        const batch_dim<2>& batch_size)
        : EnableAbstractPolymorphicObject<BatchLinOp>(exec), size_{batch_size}
    {}

    /**
     * Creates a batch operator storing items of uniform sizes.
     *
     * @param exec        the executor where all the operations are performed
     * @param num_batch_items the number of batch items to be stored in the
     * operator
     * @param size        the common size of the items in the batched operator
     */
    explicit BatchLinOp(std::shared_ptr<const Executor> exec,
                        const size_type num_batch_items = 0,
                        const dim<2>& common_size = dim<2>{})
        : BatchLinOp{std::move(exec),
                     num_batch_items > 0
                         ? batch_dim<2>(num_batch_items, common_size)
                         : batch_dim<2>{}}
    {}

private:
    batch_dim<2> size_{};
};


/**
 * A BatchLinOpFactory represents a higher order mapping which transforms one
 * batch linear operator into another.
 *
 * In a similar fashion to LinOps, BatchLinOps are also "generated" from the
 * BatchLinOpFactory. A function of this class is to provide a generate method,
 * which internally cals the generate_impl(), which the concrete BatchLinOps
 * have to implement.
 *
 * Example: using BatchCG in Ginkgo
 * ---------------------------
 *
 * ```c++
 * // Suppose A is a batch matrix, batch_b, a batch rhs vector, and batch_x, an
 * // initial guess
 * // Create a BatchCG which runs for at most 1000 iterations, and stops after
 * // reducing the residual norm by 6 orders of magnitude
 * auto batch_cg_factory = solver::BatchCg<>::build()
 *     .with_max_iters(1000)
 *     .with_rel_residual_goal(1e-6)
 *     .on(cuda);
 * // create a batch linear operator which represents the solver
 * auto batch_cg = batch_cg_factory->generate(A);
 * // solve the system
 * batch_cg->apply(gko::lend(batch_b), gko::lend(batch_x));
 * ```
 *
 * @ingroup BatchLinOp
 */
class BatchLinOpFactory
    : public AbstractFactory<BatchLinOp, std::shared_ptr<const BatchLinOp>> {
public:
    using AbstractFactory<BatchLinOp,
                          std::shared_ptr<const BatchLinOp>>::AbstractFactory;

    std::unique_ptr<BatchLinOp> generate(
        std::shared_ptr<const BatchLinOp> input) const
    {
        this->template log<log::Logger::batch_linop_factory_generate_started>(
            this, input.get());
        const auto exec = this->get_executor();
        std::unique_ptr<BatchLinOp> generated;
        if (input->get_executor() == exec) {
            generated = this->AbstractFactory::generate(input);
        } else {
            generated =
                this->AbstractFactory::generate(gko::clone(exec, input));
        }
        this->template log<log::Logger::batch_linop_factory_generate_completed>(
            this, input.get(), generated.get());
        return generated;
    }
};


/**
 * The EnableBatchLinOp mixin can be used to provide sensible default
 * implementations of the majority of the BatchLinOp and PolymorphicObject
 * interface.
 *
 * The goal of the mixin is to facilitate the development of new BatchLinOp, by
 * enabling the implementers to focus on the important parts of their operator,
 * while the library takes care of generating the trivial utility functions.
 * The mixin will provide default implementations for the entire
 * PolymorphicObject interface, including a default implementation of
 * `copy_from` between objects of the new BatchLinOp type.
 *
 * Implementers of new BatchLinOps are required to specify only the following
 * aspects:
 *
 * 1.  Creation of the BatchLinOp: This can be facilitated via either
 *     EnableCreateMethod mixin (used mostly for matrix formats),
 *     or GKO_ENABLE_BATCH_LIN_OP_FACTORY macro (used for operators created from
 *     other operators, like preconditioners and solvers).
 *
 * @tparam ConcreteBatchLinOp  the concrete BatchLinOp which is being
 *                             implemented [CRTP parameter]
 * @tparam PolymorphicBase  parent of ConcreteBatchLinOp in the polymorphic
 *                          hierarchy, has to be a subclass of BatchLinOp
 *
 * @ingroup BatchLinOp
 */
template <typename ConcreteBatchLinOp, typename PolymorphicBase = BatchLinOp>
class EnableBatchLinOp
    : public EnablePolymorphicObject<ConcreteBatchLinOp, PolymorphicBase>,
      public EnablePolymorphicAssignment<ConcreteBatchLinOp> {
public:
    using EnablePolymorphicObject<ConcreteBatchLinOp,
                                  PolymorphicBase>::EnablePolymorphicObject;
};


/**
 * This is an alias for the EnableDefaultFactory mixin, which correctly sets the
 * template parameters to enable a subclass of BatchLinOpFactory.
 *
 * @tparam ConcreteFactory  the concrete factory which is being implemented
 *                          [CRTP parameter]
 * @tparam ConcreteLinOp  the concrete BatchLinOp type which this factory
 * produces, needs to have a constructor which takes a const ConcreteFactory *,
 * and an std::shared_ptr<const BatchLinOp> as parameters.
 * @tparam ParametersType  a subclass of enable_parameters_type template which
 *                         defines all of the parameters of the factory
 * @tparam PolymorphicBase  parent of ConcreteFactory in the polymorphic
 *                          hierarchy, has to be a subclass of LinOpFactory
 *
 * @ingroup BatchLinOp
 */
template <typename ConcreteFactory, typename ConcreteBatchLinOp,
          typename ParametersType, typename PolymorphicBase = BatchLinOpFactory>
using EnableDefaultBatchLinOpFactory =
    EnableDefaultFactory<ConcreteFactory, ConcreteBatchLinOp, ParametersType,
                         PolymorphicBase>;


/**
 * This macro will generate a default implementation of a BatchLinOpFactory for
 * the BatchLinOp subclass it is defined in.
 *
 * It is required to first call the macro #GKO_CREATE_FACTORY_PARAMETERS()
 * before this one in order to instantiate the parameters type first.
 *
 * The list of parameters for the factory should be defined in a code block
 * after the macro definition, and should contain a list of
 * GKO_FACTORY_PARAMETER_* declarations. The class should provide a constructor
 * with signature
 * _batch_lin_op(const _factory_name *, std::shared_ptr<const BatchLinOp>)
 * which the factory will use a callback to construct the object.
 *
 * A minimal example of a batch linear operator is the following:
 *
 * ```c++
 * struct MyBatchLinOp : public EnableBatchLinOp<MyBatchLinOp> {
 *     GKO_ENABLE_BATCH_LIN_OP_FACTORY(MyBatchLinOp, my_parameters, Factory) {
 *         // a factory parameter named "my_value", of type int and default
 *         // value of 5
 *         int GKO_FACTORY_PARAMETER_SCALAR(my_value, 5);
 *         // a factory parameter named `my_pair` of type `std::pair<int,int>`
 *         // and default value {5, 5}
 *         std::pair<int, int> GKO_FACTORY_PARAMETER_VECTOR(my_pair, 5, 5);
 *     };
 *     // constructor needed by EnableBatchLinOp
 *     explicit MyBatchLinOp(std::shared_ptr<const Executor> exec) {
 *         : EnableBatchLinOp<MyBatchLinOp>(exec) {}
 *     // constructor needed by the factory
 *     explicit MyBatchLinOp(const Factory *factory,
 *                      std::shared_ptr<const BatchLinOp> matrix)
 *         : EnableBatchLinOp<MyBatchLinOp>(factory->get_executor()),
 *                                          matrix->get_size()),
 *           // store factory's parameters locally
 *           my_parameters_{factory->get_parameters()}
 *     {
 *          int value = my_parameters_.my_value;
 *          // do something with value
 *     }
 * ```
 *
 * MyBatchLinOp can then be created as follows:
 *
 * ```c++
 * auto exec = gko::ReferenceExecutor::create();
 * // create a factory with default `my_value` parameter
 * auto fact = MyBatchLinOp::build().on(exec);
 * // create a operator using the factory:
 * auto my_op = fact->generate(gko::batch::matrix::Identity::create(exec, 2));
 * std::cout << my_op->get_my_parameters().my_value;  // prints 5
 *
 * // create a factory with custom `my_value` parameter
 * auto fact = MyLinOp::build().with_my_value(0).on(exec);
 * // create a operator using the factory:
 * auto my_op = fact->generate(gko::batch::matrix::Identity::create(exec, 2));
 * std::cout << my_op->get_my_parameters().my_value;  // prints 0
 * ```
 *
 * @note It is possible to combine both the #GKO_CREATE_FACTORY_PARAMETER_*()
 * macros with this one in a unique macro for class __templates__ (not with
 * regular classes). Splitting this into two distinct macros allows to use them
 * in all contexts. See <https://stackoverflow.com/q/50202718/9385966> for more
 * details.
 *
 * @param _lin_op  concrete operator for which the factory is to be created
 *                 [CRTP parameter]
 * @param _parameters_name  name of the parameters member in the class
 *                          (its type is `<_parameters_name>_type`, the
 *                          protected member's name is `<_parameters_name>_`,
 *                          and the public getter's name is
 *                          `get_<_parameters_name>()`)
 * @param _factory_name  name of the generated factory type
 *
 * @ingroup BatchLinOp
 */
#define GKO_ENABLE_BATCH_LIN_OP_FACTORY(_batch_lin_op, _parameters_name,     \
                                        _factory_name)                       \
public:                                                                      \
    const _parameters_name##_type& get_##_parameters_name() const            \
    {                                                                        \
        return _parameters_name##_;                                          \
    }                                                                        \
                                                                             \
    class _factory_name                                                      \
        : public ::gko::batch::EnableDefaultBatchLinOpFactory<               \
              _factory_name, _batch_lin_op, _parameters_name##_type> {       \
        friend class ::gko::EnablePolymorphicObject<                         \
            _factory_name, ::gko::batch::BatchLinOpFactory>;                 \
        friend class ::gko::enable_parameters_type<_parameters_name##_type,  \
                                                   _factory_name>;           \
        explicit _factory_name(std::shared_ptr<const ::gko::Executor> exec)  \
            : ::gko::batch::EnableDefaultBatchLinOpFactory<                  \
                  _factory_name, _batch_lin_op, _parameters_name##_type>(    \
                  std::move(exec))                                           \
        {}                                                                   \
        explicit _factory_name(std::shared_ptr<const ::gko::Executor> exec,  \
                               const _parameters_name##_type& parameters)    \
            : ::gko::batch::EnableDefaultBatchLinOpFactory<                  \
                  _factory_name, _batch_lin_op, _parameters_name##_type>(    \
                  std::move(exec), parameters)                               \
        {}                                                                   \
    };                                                                       \
    friend ::gko::batch::EnableDefaultBatchLinOpFactory<                     \
        _factory_name, _batch_lin_op, _parameters_name##_type>;              \
                                                                             \
                                                                             \
private:                                                                     \
    _parameters_name##_type _parameters_name##_;                             \
                                                                             \
public:                                                                      \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


}  // namespace batch
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_BATCH_LIN_OP_HPP_
