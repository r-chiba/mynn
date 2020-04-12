#ifndef MYNN_OPTIMIZER_HPP_
#define MYNN_OPTIMIZER_HPP_

#include "vector.hpp"
#include "network.hpp"

namespace mynn {

class Optimizer {
protected:
    Network &network_;
    Real learningRate_;

public:
    Optimizer(Network &network, Real learningRate):
        network_(network),
        learningRate_(learningRate) {}

    virtual ~Optimizer() {}

    virtual void update() = 0;
};

} // namespace mynn

#endif /* MYNN_OPTIMIZER_HPP_ */
