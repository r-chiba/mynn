#ifndef MYNN_ACTIVATION_FUNCTION_TANH_HPP_
#define MYNN_ACTIVATION_FUNCTION_TANH_HPP_

#include "vector.hpp"
#include "activation_function.hpp"
#include <cmath>

namespace mynn {

class Tanh: public ActivationFunction {
public:
    VectorReal activate(VectorReal &x) {
        auto y = x.deepCopy();
        auto sz = y.size();
        auto xp = x.data();
        auto yp = y.data();
        for (int i = 0; i < sz; ++i) {
            *(yp + i) = std::tanhf(*(xp + i));
        }
        return y;
    }

    VectorReal differential(VectorReal &x) {
        auto y = x.deepCopy();
        auto sz = y.size();
        auto xp = x.data();
        auto yp = y.data();
        for (int i = 0; i < sz; ++i) {
            Real val = std::tanhf(*(xp + i));
            *(yp + i) = 1.0 - val * val;
        }
        return y;
    }
};

} // namespace mynn

#endif /* MYNN_ACTIVATION_FUNCTION_TANH_HPP_ */
