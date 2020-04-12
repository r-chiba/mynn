#ifndef MYNN_NETWORK_HPP_
#define MYNN_NETWORK_HPP_

#include "vector.hpp"
#include "layer/layer.hpp"
#include "activation_function/activation_function.hpp"
#include "loss_function/loss_function.hpp"

#include <vector>
#include <memory>
#include <functional>

namespace mynn {

class Network {
private:
    std::vector<std::unique_ptr<Layer> > layers_;
    std::unique_ptr<LossFunction> loss_;

public:
    Network(LossFunction *loss): loss_(loss) {}

    std::vector<std::unique_ptr<Layer> > &layers() { return layers_; }

    void addLayer(Layer *layer) {
        layers_.push_back(std::unique_ptr<Layer>(layer));
    }

    VectorReal forward(const VectorReal &x) {
        VectorReal y = x;
        int nLayers = layers_.size();
        for (int i = 0; i < nLayers; ++i) {
            y = layers_[i]->forward(y);
        }
        return y;
    }

    Real loss(const VectorReal &x, const VectorReal &label) {
        assert(x.shape() == label.shape());
        return loss_->forward(x, label);
    }

    void backward(const VectorReal &x, const VectorReal &label) {
        int nLayers = layers_.size();
        VectorReal in = layers_[nLayers-1]->activation().forward(layers_[nLayers-1]->output());
        VectorReal dout = loss_->backward(in, label);
        //std::function<Real(void)> f = [&]() { return this->loss(this->forward(x), label); };
        for (int i = nLayers - 1; i >= 0; --i) {
            in = (i > 0 ? layers_[i-1]->activation().forward(layers_[i-1]->output()) : x);
            dout = layers_[i]->activation().backward(dout);
            dout = layers_[i]->backward(in, dout, (i > 0 ? layers_[i-1].get() : nullptr));
            //layers_[i]->numericalGradient(f);
            //layers_[i]->showGradDiff();
        }
    }
};

} // namespace mynn

#endif /* MYNN_NETWORK_HPP_ */
