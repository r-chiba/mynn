#ifndef MYNN_ACTIVATION_FUNCTION_LOGISTIC_HPP_
#define MYNN_ACTIVATION_FUNCTION_LOGISTIC_HPP_

#include "vector.hpp"
#include "activation_function.hpp"
#include <cmath>

namespace mynn {

class Logistic: public ActivationFunction {
public:
    VectorReal activate(VectorReal &x) {
        auto y = x.deepCopy();
        auto sz = y.size();
        auto xp = x.data();
        auto yp = y.data();
        for (int i = 0; i < sz; ++i) {
            *(yp + i) = 1.0 / (1.0 + std::expf(-*(xp + i)));
        }
        return y;
    }

    VectorReal differential(VectorReal &x) {
        auto y = x.deepCopy();
        auto sz = y.size();
        auto xp = x.data();
        auto yp = y.data();
        for (int i = 0; i < sz; ++i) {
            Real val = 1.0 / (1.0 + std::expf(-*(xp + i)));
            *(yp + i) = val * (1.0 - val);
        }
        return y;
    }
};

} // namespace mynn

#endif /* MYNN_ACTIVATION_FUNCTION_LOGISTIC_HPP_ */
