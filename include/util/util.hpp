#ifndef MYNN_UTIL_HPP_
#define MYNN_UTIL_HPP_

#include "vector.hpp"
#include <cmath>

namespace mynn {

VectorReal Softmax(const VectorReal &x) {
    assert(x.rank() == 2);
    int batchSize = x.shape()[0];
    int dim = x.shape()[1];
    VectorReal ret(x.shape(), VectorInit::None);
    for (int i = 0; i < batchSize; ++i) {
        Real maxv = x.at({i, 0});
        for (int j = 1; j < dim; ++j) {
            maxv = std::max(maxv, x.at({i, j}));
        }
        Real sum = 0;
        for (int j = 0; j < dim; ++j) {
            sum += std::expf(x.at({i, j}) - maxv);
        }
        for (int j = 0; j < dim; ++j) {
            ret[i][j] = std::expf(x.at({i, j}) - maxv) / sum;
        }
    }
    return ret;
}

VectorReal LogSumExp(const VectorReal &x) {
    assert(x.rank() == 2);
    int batchSize = x.shape()[0];
    int dim = x.shape()[1];
    VectorReal ret({batchSize}, VectorInit::None);
    for (int i = 0; i < batchSize; ++i) {
        Real maxv = x.at({i, 0});
        for (int j = 1; j < dim; ++j) {
            maxv = std::max(maxv, x.at({i, j}));
        }
        Real sum = 0;
        for (int j = 0; j < dim; ++j) {
            sum += std::expf(x.at({i, j}) - maxv);
        }
        ret[i] = std::log(sum) + maxv;
    }
    return ret;
}

#define IN_RANGE(i, l, r) (((l) <= (i)) && ((i) < (r)))
// reshape the images to matrix for convolution
// a.k.a. im2col or lowering
// assume that filter shape is {outChannels, fh_*fw_*inChannels}
// input: images {batchSize, height, width, inChannels}
VectorReal imagesToMatrix(
        const VectorReal &images,
        int fh,
        int fw,
        int stride,
        int padding) {
    assert(images.rank() == 4);
    VectorShape imgShape = images.shape();
    int batchSize = imgShape[0];
    int imgh = imgShape[1];
    int imgw = imgShape[2];
    int imgc = imgShape[3];
    int nhBase = imgh - fh + 1 + 2*padding;
    int nwBase = imgw - fw + 1 + 2*padding;
    int nh = nhBase / stride + (nhBase % stride > 0 ? 1 : 0);
    int nw = nwBase / stride + (nwBase % stride > 0 ? 1 : 0);
    VectorReal mat({fh*fw*imgc, batchSize*nh*nw});
    int outh = 0, outw = 0;
    for (int i = 0; i < batchSize; ++i) {
        for (int h = 0-padding; h <= imgh+padding-fh; h += stride) {
            for (int w = 0-padding; w <= imgw+padding-fw; w += stride) {
                for (int k = 0; k < fh; ++k) {
                    for (int l = 0; l < fw; ++l) {
                        for (int c = 0; c < imgc; ++c) {
                            mat.at({k*(fw*imgc)+l*imgc+c,
                                    i*(nh*nw)+outh*nw+outw})
                                = ((IN_RANGE(h, 0, imgh+padding-fh+1)
                                    && IN_RANGE(w, 0, imgw+padding-fw+1))
                                    ? images.at({i, h+k, w+l, c}) : 0);
                        }
                    }
                }
                ++outw;
            }
            ++outh;
            outw = 0;
        }
    }
    return mat;
}

VectorReal matrixToImages(
        const VectorReal &mat,
        const VectorShape &imgShape,
        int fh,
        int fw,
        int stride,
        int padding) {
    assert(mat.rank() == 2 && imgShape.size() == 4);
    int batchSize = imgShape[0];
    int imgh = imgShape[1];
    int imgw = imgShape[2];
    int imgc = imgShape[3];
    int nhBase = imgh - fh + 1 + 2*padding;
    int nwBase = imgw - fw + 1 + 2*padding;
    int nh = nhBase / stride + (nhBase % stride > 0 ? 1 : 0);
    int nw = nwBase / stride + (nwBase % stride > 0 ? 1 : 0);
    VectorReal images(imgShape, VectorInit::Zero);
    int outh = 0, outw = 0;
    for (int i = 0; i < batchSize; ++i) {
        for (int h = 0-padding; h <= imgh+padding-fh; h += stride) {
            for (int w = 0-padding; w <= imgw+padding-fw; w += stride) {
                for (int k = 0; k < fh; ++k) {
                    for (int l = 0; l < fw; ++l) {
                        for (int c = 0; c < imgc; ++c) {
                            if (!(IN_RANGE(h, 0, imgh+padding-fh+1)
                                  && IN_RANGE(w, 0, imgw+padding-fw+1)))
                                    continue;
                            images.at({i, h+k, w+l, c}) +=
                                mat.at({k*(fw*imgc)+l*imgc+c,
                                        i*(nh*nw)+outh*nw+outw});
                        }
                    }
                }
                ++outw;
            }
            ++outh;
            outw = 0;
        }
    }
    return images;
}


} // namespace mynn

#endif /* MYNN_UTIL_HPP_ */
