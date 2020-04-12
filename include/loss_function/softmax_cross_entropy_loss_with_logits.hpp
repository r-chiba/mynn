#ifndef MYNN_LOSS_FUNCTION_SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS_HPP_
#define MYNN_LOSS_FUNCTION_SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS_HPP_

#include "vector.hpp"
#include "loss_function.hpp"
#include "util/util.hpp"
#include <cmath>

namespace mynn {

class SoftmaxCrossEntropyLossWithLogits: public LossFunction {
public:
    Real forward(const VectorReal &x, const VectorReal &label) const {
        assert(x.rank() == 2
               && label.rank() == 2
               && x.size() == label.size());
        int batchSize = label.shape()[0];
        int nLabels = label.shape()[1];
        VectorReal xlabel = x * label;
        VectorReal logits({batchSize}, VectorInit::Zero);
        VectorReal lse = LogSumExp(x);
        Real ret = 0;
        for (int i = 0; i < batchSize; ++i) {
            for (int j = 0; j < nLabels; ++j) {
                logits.at(i) += xlabel.at({i, j});
            }
            ret += lse.at(i) - logits.at(i);
        }
        return ret / batchSize;
    }

    VectorReal backward(const VectorReal &x, const VectorReal &label) const {
        int batchSize = x.shape()[0];
        auto dout = (Softmax(x) - label) / static_cast<Real>(batchSize);
        return dout;
    }
};

} // namespace mynn

#endif /* MYNN_LOSS_FUNCTION_SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS_HPP_ */
