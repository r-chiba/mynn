#ifndef MY_NETWORK_OPTIMIZER_MOMENTUM_SGD_OPTIMIZER_HPP_
#define MY_NETWORK_OPTIMIZER_MOMENTUM_SGD_OPTIMIZER_HPP_

#include "vector.hpp"
#include "network.hpp"
#include "layer/layer.hpp"
#include "optimizer/optimizer.hpp"
#include <unordered_map>

namespace mynn {

class MomentumSgdOptimizer: public Optimizer {
private:
    Real momentum_;
    std::unordered_map<VectorReal *, VectorReal> velocities_;

public:
    MomentumSgdOptimizer(Network &network, Real learningRate, Real momentum):
        Optimizer(network, learningRate),
        momentum_(momentum)
    {
        std::vector<std::unique_ptr<Layer>> &layers = network_.layers();
        int nLayers = layers.size();
        if (velocities_.size() == 0) {
            for (int i = nLayers - 1; i >= 0 ; --i) {
                auto params = layers[i]->parameters();
                for (auto param: params) {
                    velocities_[param] = VectorReal(param->shape(), VectorInit::Zero);
                }
            }
        }
    }

    void update() {
        std::vector<std::unique_ptr<Layer>> &layers = network_.layers();
        int nLayers = layers.size();
        for (int i = nLayers - 1; i >= 0 ; --i) {
            auto params = layers[i]->parameters();
            for (auto param: params) {
                auto &v = velocities_[param];
                v = momentum_ * v - learningRate_ * layers[i]->gradient(param);
                *param += v;
            }
        }
    }
};

} // namespace mynn

#endif /* MY_NETWORK_OPTIMIZER_MOMENTUM_SGD_OPTIMIZER_HPP_ */
