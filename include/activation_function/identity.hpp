#ifndef MYNN_ACTIVATION_FUNCTION_IDENTITY_HPP_
#define MYNN_ACTIVATION_FUNCTION_IDENTITY_HPP_

#include "vector.hpp"
#include "activation_function.hpp"

namespace mynn {

class Identity: public ActivationFunction {
public:
    VectorReal forward(const VectorReal &x) {
        return x.deepCopy();
    }

    VectorReal backward(const VectorReal &x) {
        return x.deepCopy();
    }
};

} // namespace mynn

#endif /* MYNN_ACTIVATION_FUNCTION_IDENTITY_HPP_ */
