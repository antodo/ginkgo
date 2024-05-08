// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_CONFIG_CONFIG_HPP_
#define GKO_PUBLIC_CORE_CONFIG_CONFIG_HPP_


#include <map>
#include <string>
#include <unordered_map>


#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/type_descriptor.hpp>
#include <ginkgo/core/solver/solver_base.hpp>


namespace gko {
namespace config {

class registry;


class pnode;


/**
 * parse is the main entry point to create an Ginkgo object based on
 * some file configuration. It reads a configuration stored as a property tree
 * and creates the desired type.
 *
 * General rules for configuration
 * 1. all parameter and template usage are according to the class directly. It
 *    has the same behavior as the class like default setting without specifying
 *    anything. When the class factory parameters allows `with_<key>(value)`,
 *    the file configuration will allow `"<key>": value`
 * 2. all key will use snake_case including template. For example, ValueType ->
 *    value_type
 * 3. If the value is not bool, integer, or floating point, we will use string
 *    to represent everything. For example, we will use string to select the
 *    enum value. `"baseline": "absolute"` will select the absolute baseline in
 *    ResidualNorm critrion
 * 4. `"type"` is the new key to select the class without template type. We also
 *    prepend the namespace except for gko. For example, we use "solver::Cg" for
 *    Cg solver. Note. the template type is given by the another entry or from
 *    the type_descriptor.
 * 5. the data type uses fixed-width representation
 *    void, int32, int64, float32, float64, complex<float32>, complex<float64>
 * 6. We use [real, imag] to represent complex value. If it only contains one
 *    value or without [], we will treat it as the complex number with imaginary
 *    part = 0.
 * 7. In many cases, the parameter allows vector input and we handle it by array
 *    of property tree. If the array only contains one object, users can
 *    directly provide the object without putting it into an array.
 *    `"criteria": [{...}]` and `"criteria": {...}` are the same.
 *
 * The configuration needs
 * to specify the resulting type by the field:
 * ```
 * "type": "some_supported_ginkgo_type"
 * ```
 * The result will be a deferred_factory_parameter,
 * which can be thought of as an intermediate step before a LinOpFactory.
 * Providing the result an Executor through the function `.on(exec)` will then
 * create the factory with the parameters as defined in the configuration.
 *
 * Given a configuration that is defined as
 * ```
 * "type": "solver::Gmres",
 * "krylov_dim": 20,
 * "criteria": [
 *   {"type": "stop::Iteration", "max_iters": 10},
 *   {"type": "stop::ResidualNorm", "reduction_factor": 1e-6}
 * ]
 * ```
 * then passing it to this function like this:
 * ```c++
 * auto gmres_factory = parse(config, context);
 * ```
 * will create a factory for a GMRES solver, with the parameters `krylov_dim`
 * set to 20, and a combined stopping criteria, consisting of an Iteration
 * criteria with maximal 10 iterations, and a ResidualNorm criteria with a
 * reduction factor of 1e-6.
 *
 * By default, the factory will use the value type double, and index type
 * int32 when creating templated types. This can be changed by passing in a
 * type_descriptor. For example:
 * ```c++
 * auto gmres_factory =
 *     parse(config, context,
 *           make_type_descriptor<float32, int32>());
 * ```
 * will lead to a GMRES solver that uses `float` as its value type.
 * Additionally, the config can be used to set these types through the fields:
 * ```
 * value_type: "some_value_type"
 * index_type: "some_index_type"
 * ```
 * These types take precedence over the type descriptor and they are used for
 * every created object beginning from the config level they are defined on and
 * every deeper nested level, until a new type is defined. So, considering the
 * following example
 * ```
 * type: "solver::Ir",
 * value_type: "float32"
 * solver: {
 *   type: "solver::Gmres",
 *   preconditioner: {
 *     type: "preconditioner::Jacobi"
 *     value_type: "float64"
 *   }
 * }
 * ```
 * both the Ir and Gmres are using `float32` as a value type, and the
 * Jacobi uses `float64`.
 *
 * @param config  The property tree which must include `type` for the class
 *                base.
 * @param context  The registry which stores the building function map and the
 *                 storage for generated objects.
 * @param type_descriptor  The default value and index type. If any object that
 *                         is created as part of this configuration has a
 *                         templated type, then the value and/or index type from
 *                         the descriptor will be used. Any definition of the
 *                         value and/or index type within the config will take
 *                         precedence over the descriptor.
 *
 * @return a deferred_factory_parameter which creates an LinOpFactory after
 *         `.on(exec)` is called on it.
 */
deferred_factory_parameter<gko::LinOpFactory> parse(
    const pnode& config, const registry& context,
    const type_descriptor& td = make_type_descriptor<>());


}  // namespace config
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_CONFIG_CONFIG_HPP_
