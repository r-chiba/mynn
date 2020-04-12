#ifndef MYNN_LOSS_FUNCTION_HPP_
#define MYNN_LOSS_FUNCTION_HPP_

#include "vector.hpp"

namespace mynn {

class LossFunction {
public:
    virtual ~LossFunction() {}
    virtual Real forward(const VectorReal &x, const VectorReal &label) const = 0;
    virtual VectorReal backward(const VectorReal &x, const VectorReal &label) const = 0;
};

} // namespace mynn

#endif /* MYNN_LOSS_FUNCTION_HPP_ */
