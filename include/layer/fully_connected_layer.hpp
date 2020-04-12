#ifndef MYNN_LAYER_FULLY_CONNECTED_LAYER_H_
#define MYNN_LAYER_FULLY_CONNECTED_LAYER_H_

#include "vector.hpp"
#include "layer.hpp"
#include <cmath>
#include <vector>
#include <random>
#include <functional>

namespace mynn {

class FullyConnectedLayer: public Layer {
private:
    template<typename Distribution>
    void initializeVector(VectorReal &v, Distribution &dist) {
        std::random_device rd;
        std::default_random_engine e(rd());
        std::size_t sz = v.size();
        for (std::size_t i = 0; i < sz; i++) {
            v.at(i) = dist(e);
        }
    }

public:
    int dimIn_, dimOut_; // dimensions of input data and output data
    VectorReal w_; // weight matrix of the shape {dimOut_, dimIn}
    VectorReal b_; // bias vector of the shape {dimOut_}
    VectorReal gw_, gb_; // grads of the loss wrt w_ and b_
    VectorReal gwNumeric_, gbNumeric_; // grads of the loss wrt w_ and b_

    template<typename Distribution>
    FullyConnectedLayer(int dimIn,
                        int dimOut,
                        ActivationFunction *activation,
                        Distribution *dist):
        Layer(activation),
        dimIn_(dimIn),
        dimOut_(dimOut),
        w_({dimOut, dimIn}),
        b_({dimOut}),
        gw_({dimOut, dimIn}),
        gb_({dimOut}),
        gwNumeric_({dimOut, dimIn}),
        gbNumeric_({dimOut})
    {
        std::unique_ptr<Distribution> p(dist);
        initializeVector(w_, *p.get());
        //initializeVector(b_, *p.get());
        b_ = 0;
    };

    // forward propagation
    // in: input data {batchSize, dimIn_}
    // return value: activation_(w_ * in + b_)
    VectorReal forward(const VectorReal &in) {
        assert(in.rank() == 2 && in.shape()[1] == dimIn_);
        int batchSize = in.shape()[0];
        output_ = matMul(in, w_.transpose());
        for (int i = 0; i < batchSize; ++i) {
            output_[i] += b_;
        }
        return activation_->forward(output_);
    }

    // backward propagation
    // in: input data of this->forward() {batchSize, dimIn_}
    // dout: gradient of loss wrt output of this layer {batchSize, dimOut_}
    // prev: previous layer (if exist)
    // return value: gradient of loss wrt input of this layer {batchSize, dimIn_}
    VectorReal backward(const VectorReal &in, const VectorReal &dout, const Layer *prev) {
        assert(in.rank() == 2
               && in.shape()[1] == dimIn_
               && dout.rank() == 2
               && dout.shape()[1] == dimOut_);
        int batchSize = in.shape()[0];
        gw_ = matMul(dout.transpose(), in);
        gb_ = 0;
        for (int i = 0; i < batchSize; ++i) {
            gb_ += dout[i];
        }
        if (prev) {
            return matMul(dout, w_);
        } else {
            return VectorReal();
        }
    }

#if 1
    void numericalGradient(const std::function<Real(void)> &f) {
        constexpr Real h = 1e-4;
        gwNumeric_ = VectorReal(w_.shape(), VectorInit::None);
        gbNumeric_ = VectorReal(b_.shape(), VectorInit::None);
        for (int i = 0; i < w_.size(); ++i) {
            Real tmp = w_.at(i);
            w_.at(i) = tmp + h;
            Real fxh1 = f();
            w_.at(i) = tmp - h;
            Real fxh2 = f();
            gwNumeric_.at(i) = (fxh1 - fxh2) / (2*h);
            w_.at(i) = tmp;
        }
        for (int i = 0; i < b_.size(); ++i) {
            Real tmp = b_.at(i);
            b_.at(i) = tmp + h;
            Real fxh1 = f();
            b_.at(i) = tmp - h;
            Real fxh2 = f();
            gbNumeric_.at(i) = (fxh1 - fxh2) / (2*h);
            b_.at(i) = tmp;
        }
    }

    void showGradDiff() {
        Real wdiff = 0, bdiff = 0;
        for (int i = 0; i < gw_.size(); ++i) {
            wdiff += std::abs(gw_.at(i) - gwNumeric_.at(i));
        }
        wdiff /= gw_.size();
        for (int i = 0; i < gb_.size(); ++i) {
            bdiff += std::abs(gb_.at(i) - gbNumeric_.at(i));
        }
        bdiff /= gb_.size();
        std::cerr << "gw: " << wdiff << std::endl;
        //std::cerr << gw_ << std::endl;
        //std::cerr << gwNumeric_ << std::endl;
        //std::cerr << gw_ - gwNumeric_ << std::endl;
        std::cerr << "gb: " << bdiff << std::endl;
        //std::cerr << gb_ << std::endl;
        //std::cerr << gbNumeric_ << std::endl;
        //std::cerr << gb_ - gbNumeric_ << std::endl;
    }
#endif

    std::vector<VectorReal *> parameters() {
        return { &w_, &b_ };
    }

    const VectorReal &gradient(VectorReal *parameter) {
        assert(parameter == &w_ || parameter == &b_);
        if (parameter == &w_) return gw_;
        else return gb_;
    }
};

} // namespace mynn

#endif /* MYNN_LAYER_FULLY_CONNECTED_LAYER_H_ */
