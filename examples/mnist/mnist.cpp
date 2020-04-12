#include "vector.hpp"
#include "network.hpp"
#include "layer/fully_connected_layer.hpp"
#include "activation_function/identity.hpp"
#include "activation_function/relu.hpp"
#include "loss_function/softmax_cross_entropy_loss_with_logits.hpp"
#include "optimizer/sgd_optimizer.hpp"
#include "optimizer/momentum_sgd_optimizer.hpp"

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdint>
#include <iostream>


using namespace std;
using namespace mynn;

static string mnistImageFiles[] = {
    "train-images-idx3-ubyte"s,
    "t10k-images-idx3-ubyte"s,
};
static string mnistLabelFiles[] = {
    "train-labels-idx1-ubyte"s,
    "t10k-labels-idx1-ubyte"s,
};

static uint32_t swap32Bytes(uint32_t value) {
    uint32_t ret = 0;
    ret |= ((value >>  0) & 0xff) << 24;
    ret |= ((value >>  8) & 0xff) << 16;
    ret |= ((value >> 16) & 0xff) <<  8;
    ret |= ((value >> 24) & 0xff) <<  0;
    return ret;
}

static bool mapFile(const char *filePath, uint8_t **buf, size_t *size) {
    bool ret = true;
    int fd = -1, res;
    struct stat st;

    if (filePath == nullptr || buf == nullptr || size == nullptr) {
        return false;
    }

    fd = open(filePath, O_RDONLY);
    if (fd == -1) {
        fprintf(stderr, "open(%s) failed. errno=%d\n", filePath, errno);
        ret = false;
        goto end;
    }
    res = fstat(fd, &st);
    if (res == -1) {
        fprintf(stderr, "fstat(%s) failed. errno=%d\n", filePath, errno);
        ret = false;
        goto end;
    }
    *buf = static_cast<uint8_t *>(mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (buf == MAP_FAILED) {
        fprintf(stderr, "mmap(%s) failed. errno=%d\n", filePath, errno);
        ret = false;
        goto end;
    }
    *size = st.st_size;

end:
    if (fd != -1) close(fd);
    return ret;
}

static void unmapFile(uint8_t *buf, size_t size) {
    if (buf == nullptr || buf == MAP_FAILED) {
        return;
    }
    munmap(buf, size);
}

static bool getImageData(
        const uint8_t *buf,
        VectorReal &images) {
    if (buf == nullptr || buf == MAP_FAILED) {
        return false;
    }
    const uint8_t *p = buf;
    int nImages, nRows, nCols;
    if (*reinterpret_cast<const uint32_t *>(p) != 0x03080000) {
        fprintf(stderr, "magic not matched\n");
        return false;
    }
    p += sizeof(uint32_t);
    nImages = static_cast<int>(swap32Bytes(*reinterpret_cast<const uint32_t *>(p)));
    p += sizeof(uint32_t);
    nRows = static_cast<int>(swap32Bytes(*reinterpret_cast<const uint32_t *>(p)));
    p += sizeof(uint32_t);
    nCols = static_cast<int>(swap32Bytes(*reinterpret_cast<const uint32_t *>(p)));
    p += sizeof(uint32_t);
    //fprintf(stderr, "%d %d %d\n", nImages, nRows, nCols);
    images = VectorReal({nImages, nRows*nCols}, VectorInit::None);
    for (int i = 0; i < nImages; ++i) {
        vector<Real> image;
        for (int j = 0; j < nRows*nCols; ++j) {
            image.push_back(static_cast<Real>(*p++));
        }
        images[i] = VectorReal(image, {nRows*nCols});
    }
    return true;
}

static bool getLabelData(
        const uint8_t *buf,
        VectorReal &labels) {
    if (buf == nullptr || buf == MAP_FAILED) {
        return false;
    }
    const uint8_t *p = buf;
    int nLabels;
    if (*reinterpret_cast<const uint32_t *>(p) != 0x01080000) {
        fprintf(stderr, "magic not matched\n");
        return false;
    }
    p += sizeof(uint32_t);
    nLabels = static_cast<int>(swap32Bytes(*reinterpret_cast<const uint32_t *>(p)));
    p += sizeof(uint32_t);
    //fprintf(stderr, "%u\n", nLabels);
    labels = VectorReal({nLabels, 10}, VectorInit::Zero);
    for (int i = 0; i < nLabels; ++i) {
        uint8_t l = *p++;
        labels[i][l] = 1;
    }
    return true;
}

int main(int argc, char *argv[]) {
    char filePath[1024];
    VectorReal trainImages, testImages;
    VectorReal trainLabels, testLabels;
    VectorReal *images[] = {&trainImages, &testImages};
    VectorReal *labels[] = {&trainLabels, &testLabels};

    if (argc < 2) {
        fprintf(stderr, "[USAGE] %s path/to/mnist/data/dir\n", argv[0]);
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < sizeof(mnistImageFiles)/sizeof(mnistImageFiles[0]); ++i) {
        uint8_t *buf;
        size_t size;
        int written = snprintf(filePath, sizeof(filePath),
                                "%s/%s", argv[1], mnistImageFiles[i].c_str());
        if (written < 0) {
            fprintf(stderr, "snprintf() failed.\n");
        }
        fprintf(stderr, "extracting %s...\n", filePath);
        bool ret = mapFile(filePath, &buf, &size);
        if (!ret) {
            fprintf(stderr, "mapFile(%s) failed.\n", filePath);
            return EXIT_FAILURE;
        }
        ret = getImageData(buf, *images[i]);
        if (!ret) {
            fprintf(stderr, "getImageData(%s) failed.\n", filePath);
            unmapFile(buf, size);
            return EXIT_FAILURE;
        }
        unmapFile(buf, size);
    }

    for (size_t i = 0; i < sizeof(mnistLabelFiles)/sizeof(mnistLabelFiles[0]); ++i) {
        uint8_t *buf;
        size_t size;
        int written = snprintf(filePath, sizeof(filePath),
                                "%s/%s", argv[1], mnistLabelFiles[i].c_str());
        if (written < 0) {
            fprintf(stderr, "snprintf() failed.\n");
        }
        fprintf(stderr, "extracting %s...\n", filePath);
        bool ret = mapFile(filePath, &buf, &size);
        if (!ret) {
            fprintf(stderr, "mapFile(%s) failed.\n", filePath);
            return EXIT_FAILURE;
        }
        ret = getLabelData(buf, *labels[i]);
        if (!ret) {
            fprintf(stderr, "getLabelData(%s) failed.\n", filePath);
            unmapFile(buf, size);
            return EXIT_FAILURE;
        }
        unmapFile(buf, size);
    }
    fprintf(stderr, "done.\n");

    fprintf(stderr, "creating the network...\n");
    Network net(new SoftmaxCrossEntropyLossWithLogits());
    net.addLayer(new FullyConnectedLayer(
                        28*28,
                        100,
                        new Relu(),
                        new std::normal_distribution<Real>{0, 0.01}));
    net.addLayer(new FullyConnectedLayer(
                        100,
                        10,
                        new Identity(),
                        new std::normal_distribution<Real>{0, 0.01}));
    MomentumSgdOptimizer opt(net, 0.1, 0.9);
    fprintf(stderr, "done.\n");

    int nEpochs = 5;
    int batchSize = 300;
    fprintf(stderr, "start training...\n");
    int nTrainImages = trainImages.shape()[0];
    int nTestImages = testImages.shape()[0];
    trainImages /= 255;
    trainImages -= 0.5;
    testImages /= 255;
    testImages -= 0.5;
    for (int epoch = 0; epoch < nEpochs; ++epoch) {
        fprintf(stderr, "epoch %d...\n", epoch+1);
        vector<int> indices;
        //cerr << "shuffling..." << endl;
        //trainImages.shuffle(&indices);
        //trainLabels.shuffle(&indices);
        //cerr << "done." << endl;
        for (int i = 0; i < nTrainImages; i += batchSize) {
            fprintf(stderr, "batch %d...\n", i/batchSize+1);
            for (int j = 0; j < 28; ++j) {
                for (int k = 0; k < 28; ++k) {
                    fprintf(stderr, "%d", (trainImages[i].at(j*28+k) > 0 ? 1 : 0));
                }
                cout << endl;
            }
            cout << trainLabels[i] << endl;
            uint64_t s, e;
            s = time(NULL);
            auto pred = net.forward(trainImages(i, i+batchSize));
            auto loss = net.loss(pred, trainLabels(i, i+batchSize));
            net.backward(trainImages(i, i+batchSize), trainLabels(i, i+batchSize));
            opt.update();
            e = time(NULL);
            cerr << e - s << endl;
            double correct = 0;
            for (int j = 0; j < batchSize; ++j) {
                int mp = INT_MIN, mpi = -1, mli = -1;
                for (int k = 0; k < 10; ++k) {
                    if (mpi == -1 || mp < pred.at({j, k})) {
                        mp = pred.at({j, k});
                        mpi = k;
                    }
                    if (trainLabels(i, i+batchSize).at({j, k}) > 0) {
                        mli = k;
                    }
                }
                if (mpi == mli) correct += 1;
            }
            double acc = correct / batchSize * 100;
            cout << "acc=" << acc << endl;
            cout << "loss=" << loss << endl;
        }

        double correct = 0;
        auto pred = net.forward(testImages);
        for (int i = 0; i < nTestImages; ++i) {
            int mp = INT_MIN, mpi = -1, mli = -1;
            for (int j = 0; j < 10; ++j) {
                if (mpi == -1 || mp < pred.at({i, j})) {
                    mp = pred.at({i, j});
                    mpi = j;
                }
                if (testLabels.at({i, j}) > 0) {
                    mli = j;
                }
            }
            if (mpi == mli) correct += 1;
        }
        double acc = correct / nTestImages * 100;
        cout << "test acc=" << acc << endl;
    }
    return EXIT_SUCCESS;
}
