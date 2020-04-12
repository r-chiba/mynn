#ifndef MYNN_ACTIVATION_FUNCTION_HPP_
#define MYNN_ACTIVATION_FUNCTION_HPP_

#include "vector.hpp"

namespace mynn {

class ActivationFunction {
public:
    virtual ~ActivationFunction() {}
    virtual VectorReal forward(const VectorReal &x) = 0;
    virtual VectorReal backward(const VectorReal &x) = 0;
};

} // namespace mynn

#endif /* MYNN_ACTIVATION_FUNCTION_HPP_ */
