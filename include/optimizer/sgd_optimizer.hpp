#ifndef MYNN_OPTIMIZER_SGD_OPTIMIZER_HPP_
#define MYNN_OPTIMIZER_SGD_OPTIMIZER_HPP_

#include "vector.hpp"
#include "network.hpp"
#include "layer/layer.hpp"
#include "optimizer/optimizer.hpp"

namespace mynn {

class SgdOptimizer: public Optimizer {
public:
    SgdOptimizer(Network &network, Real learningRate):
        Optimizer(network, learningRate) {}

    void update() {
        std::vector<std::unique_ptr<Layer>> &layers = network_.layers();
        int nLayers = layers.size();
        for (int i = nLayers - 1; i >= 0 ; --i) {
            auto params = layers[i]->parameters();
            for (auto param: params) {
                *param -= learningRate_ * layers[i]->gradient(param);
            }
        }
    }
};

} // namespace mynn

#endif /* MYNN_OPTIMIZER_SGD_OPTIMIZER_HPP_ */
