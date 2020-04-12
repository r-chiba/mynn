#ifndef MYNN_LAYER_HPP_
#define MYNN_LAYER_HPP_

#include "vector.hpp"
#include "activation_function/activation_function.hpp"
#include <vector>
#include <memory>
#include <functional>

namespace mynn {

class Layer {
protected:
    VectorReal output_; // output of the layer (not activated)
    std::unique_ptr<ActivationFunction> activation_; // activation function

public:
    Layer(ActivationFunction *activation):
        activation_(activation) {}
    virtual ~Layer() {}
    const VectorReal &output() const { return output_; }
    ActivationFunction &activation() { return *activation_.get(); }

    virtual VectorReal forward(const VectorReal &in) = 0;
    virtual VectorReal backward(const VectorReal &in,
                                const VectorReal &dout,
                                const Layer *prev) = 0;
    virtual void showGradDiff() = 0;
    virtual void numericalGradient(const std::function<Real(void)> &f) = 0;

    virtual std::vector<VectorReal *> parameters() = 0;
    virtual const VectorReal &gradient(VectorReal *parameter) = 0;
};

} // namespace mynn

#endif /* MYNN_LAYER_HPP_ */
