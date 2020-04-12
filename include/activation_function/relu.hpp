#ifndef MYNN_ACTIVATION_FUNCTION_RELU_HPP_
#define MYNN_ACTIVATION_FUNCTION_RELU_HPP_

#include "vector.hpp"
#include "activation_function.hpp"

namespace mynn {

class Relu: public ActivationFunction {
private:
    std::vector<int> indices_;

public:
    VectorReal forward(const VectorReal &x) {
        indices_.clear();
        auto y = x.deepCopy();
        auto sz = y.size();
        for (int i = 0; i < sz; ++i) {
            if (x.at(i) < 0) {
                y.at(i) = 0;
                indices_.push_back(i);
            }
        }
        return y;
    }

    VectorReal backward(const VectorReal &x) {
        auto y = x.deepCopy();
        for (int i: indices_) {
            y.at(i) = 0;
        }
        return y;
    }
};

} // namespace mynn

#endif /* MYNN_ACTIVATION_FUNCTION_RELU_HPP_ */
