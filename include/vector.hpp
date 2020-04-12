#ifndef MYNN_VECTOR_HPP_
#define MYNN_VECTOR_HPP_

#include <cblas.h>
#include <cstddef>
#include <cassert>
#include <memory>
#include <vector>
#include <initializer_list>
#include <random>
#include <iostream>
#include <ios>
#include <iomanip>

namespace mynn {

using VectorShape = std::vector<int>;

std::ostream &operator<<(std::ostream &os, const VectorShape &v) {
    std::size_t sz = v.size();
    if (sz == 0) {
        os << "empty";
    } else {
        os << "{";
        for (std::size_t i = 0; i < sz-1; i++) {
            os << v[i] << ", ";
        }
        os << v[sz-1] <<  "}";
    }
    return os;
}

enum class VectorInit {
    None,
    Zero,
    One,
};

// n-dimensional array like numpy's ndarray
template<typename T>
class Vector {
private:
    std::shared_ptr<T> data_;
    T *view_;
    VectorShape shape_;
    VectorShape strides_;
    int size_;
    bool isTransposed_;

    static int GetSize(const VectorShape &shape) {
        if (shape.size() == 0) return 0;
        int sz = 1;
        for (auto s: shape) {
            sz *= s;
        }
        return sz;
    }

    static VectorShape ConstructStrides(const VectorShape &shape) {
        VectorShape strides;
        if (shape.size() == 0) return strides;
        strides.push_back(1);
        if (shape.size() > 1) {
            int last = 1;
            for (auto it = shape.rbegin(); it != std::prev(shape.rend()); ++it) {
                last *= *it;
                strides.push_back(last);
            }
        }
        std::reverse(strides.begin(), strides.end());
        return strides;
    }

    Vector(T *data,
           const VectorShape &shape,
           const VectorShape &strides,
           bool isTransposed=false):
        data_(nullptr),
        view_(data),
        shape_(shape),
        strides_(strides),
        size_(GetSize(shape)),
        isTransposed_(isTransposed) {}

    Vector(T *data, const VectorShape &shape, bool isTransposed=false):
        Vector<T>(data, shape, ConstructStrides(shape), isTransposed) {}


    T *data() {
        if (view_) return view_;
        return data_.get();
    }

    template<class BinaryOperation>
    void applyOperatorElementWise(const Vector<T> &other, BinaryOperation op) & {
        assert(shape_ == other.shape());
        int sz = size();
        for (int i = 0; i < sz; ++i) {
            at(i) = op(at(i), other.at(i));
        }
    }

    template<class BinaryOperation>
    void applyOperatorElementWise(T val, BinaryOperation op) & {
        int sz = size();
        for (int i = 0; i < sz; ++i) {
            at(i) = op(at(i), val);
        }
    }

public:
    Vector():
        data_(nullptr),
        view_(nullptr),
        size_(0),
        isTransposed_(false) {}
    
    // reshaped shallow copy
    Vector(const Vector<T> &other,
           const VectorShape &newShape,
           const VectorShape &newStrides,
           bool isTransposed=false):
        data_(other.data_),
        view_(other.view_),
        shape_(newShape),
        strides_(newStrides),
        size_(other.size_),
        isTransposed_(isTransposed)
    {
        assert(other.size() == GetSize(newShape));
    }

    Vector(const Vector<T> &other, const VectorShape &newShape, bool isTransposed=false):
        Vector<T>(other, newShape, ConstructStrides(newShape), isTransposed) {}

    // shallow copy
    Vector(const Vector<T> &other, bool isTransposed=false):
        Vector<T>(other, other.shape_, other.strides_, isTransposed)
    {
        if (isTransposed != other.isTransposed_) {
            std::reverse(shape_.begin(), shape_.end());
            std::reverse(strides_.begin(), strides_.end());
        }
    };

    Vector(Vector<T> &&other):
        data_(std::move(other.data_)),
        view_(std::move(other.view_)),
        shape_(std::move(other.shape_)),
        strides_(std::move(other.strides_)),
        size_(std::move(other.size_)),
        isTransposed_(std::move(other.isTransposed_))
    {
        other.data_ = std::shared_ptr<T>(nullptr);
        other.view_ = nullptr;
    }

    Vector(const VectorShape &shape,
           const VectorShape &strides,
           VectorInit init = VectorInit::None):
        view_(nullptr),
        shape_(shape),
        strides_(strides),
        size_(GetSize(shape)),
        isTransposed_(false)
    {
        if (shape_.size() > 0) {
            data_ = std::shared_ptr<T>(new T[size_], std::default_delete<T[]>());
            T *p = data_.get();
            if (init == VectorInit::Zero) {
                for (int i = 0; i < size_; ++i) p[i] = static_cast<T>(0);
            } else if (init == VectorInit::One) {
                for (int i = 0; i < size_; ++i) p[i] = static_cast<T>(1);
            }
        } else {
            data_ = std::shared_ptr<T>(nullptr);
        }
    }

    Vector(const VectorShape &shape, VectorInit init = VectorInit::None):
        Vector<T>(shape, ConstructStrides(shape), init) {}

    Vector(std::initializer_list<int> shape, VectorInit init = VectorInit::None):
        Vector<T>(VectorShape(shape.begin(), shape.end()), init) {}

    Vector(const std::vector<T> &v, const VectorShape &shape):
        view_(nullptr),
        shape_(shape),
        strides_(ConstructStrides(shape)),
        size_(v.size()),
        isTransposed_(false)
    {
        assert(size_ == GetSize(shape_));
        if (size_ > 0) {
            data_ = std::shared_ptr<T>(new T[size_], std::default_delete<T[]>());
            T *p = data_.get();
            for (int i = 0; i < size_; ++i) {
                p[i] = v[i];
            }
        } else {
            data_ = std::shared_ptr<T>(nullptr);
        }
    }

    ~Vector() {}

    const T *data() const {
        if (view_) return view_;
        return data_.get();
    }

    const VectorShape &shape() const {
        return shape_;
    }

    const VectorShape &strides() const {
        return strides_;
    }

    int size() const {
        return size_;
    }

    int rank() const {
        return shape_.size();
    }

    bool isTransposed() const {
        return isTransposed_;
    }

    Vector<T> &operator=(const Vector<T> &other) {
        if (this != &other) {
            if (shape_ == other.shape_) {
                for (int i = 0; i < size_; ++i) {
                    at(i) = other.at(i);
                }
            } else {
                data_ = other.data_;
                view_ = other.view_;
                shape_ = other.shape_;
                strides_ = other.strides_;
                size_ = other.size_;
                isTransposed_ = other.isTransposed_;
            }
        }
        return *this;
    }

    Vector<T> &operator=(Vector<T> &&other) noexcept {
        if (shape_ == other.shape_) {
            for (int i = 0; i < size_; ++i) {
                at(i) = other.at(i);
            }
        } else {
            data_ = std::move(other.data_);
            view_ = std::move(other.view_);
            shape_ = std::move(other.shape_);
            strides_ = std::move(other.strides_);
            size_ = std::move(other.size_);
            isTransposed_ = std::move(other.isTransposed_);
        }
        other.data_ = std::shared_ptr<T>(nullptr);
        other.view_ = nullptr;
        return *this;
    }

    Vector<T> &operator=(T val) {
        applyOperatorElementWise(val, [](T lhs, T rhs) { (void)lhs; return rhs; });
        return *this;
    }

    Vector<T> operator+() const {
        return *this;
    }

    Vector<T> operator-() const {
        Vector<T> ret = this->deepCopy();
        ret.applyOperatorElementWise(0, [](T lhs, T rhs) { (void)rhs; return -lhs; });
        return ret;
    }

    Vector<T> &operator+=(const Vector<T> &other) {
        applyOperatorElementWise(other, std::plus<T>());
        return *this;
    }

    Vector<T> &operator+=(T val) {
        applyOperatorElementWise(val, std::plus<T>());
        return *this;
    }

    Vector<T> &operator-=(const Vector<T> &other) {
        applyOperatorElementWise(other, std::minus<T>());
        return *this;
    }

    Vector<T> &operator-=(T val) {
        applyOperatorElementWise(val, std::minus<T>());
        return *this;
    }

    Vector<T> &operator*=(const Vector<T> &other) {
        applyOperatorElementWise(other, std::multiplies<T>());
        return *this;
    }

    Vector<T> &operator*=(T val) {
        applyOperatorElementWise(val, std::multiplies<T>());
        return *this;
    }

    Vector<T> &operator/=(const Vector<T> &other) {
        applyOperatorElementWise(other, std::divides<T>());
        return *this;
    }

    Vector<T> &operator/=(T val) {
        applyOperatorElementWise(val, std::divides<T>());
        return *this;
    }

    Vector<T> &operator%=(const Vector<T> &other) {
        applyOperatorElementWise(other, std::modulus<T>());
        return *this;
    }

    Vector<T> &operator%=(T val) {
        applyOperatorElementWise(val, std::modulus<T>());
        return *this;
    }

    Vector<T> operator[](int idx) {
        assert(rank() > 0 && idx < shape_[0]);
        int offset = strides_[0];
        VectorShape subShape;
        if (rank() > 1) {
            subShape = VectorShape(std::next(shape_.begin()), shape_.end());
        } else {
            subShape = {1};
        }
        VectorShape strides(std::next(strides_.begin()), strides_.end());
        return Vector<T>(data() + idx * offset, subShape, strides, isTransposed_);
    }

    const Vector<T> operator[](int idx) const {
        assert(rank() > 0 && idx < shape_[0]);
        int offset = strides_[0];
        VectorShape subShape;
        if (rank() > 1) {
            subShape = VectorShape(std::next(shape_.begin()), shape_.end());
        } else {
            subShape = {1};
        }
        VectorShape strides(std::next(strides_.begin()), strides_.end());
        return Vector<T>(const_cast<T *>(data() + idx * offset), subShape, strides, isTransposed_);
    }

    Vector<T> operator()(int sidx, int eidx) {
        assert(rank() > 0 && sidx < eidx && eidx <= shape_[0]);
        int offset = strides_[0];
        VectorShape subShape(shape_);
        subShape[0] = eidx - sidx;
        return Vector<T>(data() + sidx * offset, subShape, strides_, isTransposed_);
    }

    const Vector<T> operator()(int sidx, int eidx) const {
        assert(rank() > 0 && sidx < eidx && eidx <= shape_[0]);
        int offset = strides_[0];
        VectorShape subShape(shape_);
        subShape[0] = eidx - sidx;
        return Vector<T>(const_cast<T *>(data() + sidx * offset), subShape, strides, isTransposed_);
    }

    int toIndex(const VectorShape &indices) const {
        assert(indices.size() == static_cast<std::size_t>(rank()));
        int idx = 0, i = 0;
        for (auto it = indices.begin(); it != indices.end(); ++it) {
            if (*it >= shape_[i]) {
                std::cerr << shape_ << " " << indices << " " << i << " " << *it << std::endl;
            }
            assert(*it < shape_[i]);
            idx += *it * strides_[i++];
        }
        return idx;
    }

    int toRealIndex(int idx) const {
        assert(isTransposed_);
        int ridx = 0;
        for (int i = rank()-1; i >= 0; --i) {
            ridx += (idx % shape_[i]) * strides_[i];
            idx /= shape_[i];
        }
        return ridx;
    }

    T &at(const VectorShape &indices) {
        assert(indices.size() == static_cast<std::size_t>(rank()));
        return at(toIndex(indices));
    }

    const T &at(const VectorShape &indices) const {
        assert(indices.size() == static_cast<std::size_t>(rank()));
        return at(toIndex(indices));
    }

    T &at(int idx) {
        assert(idx < size());
        if (isTransposed_) idx = toRealIndex(idx);
        return data()[idx];
    }

    const T &at(int idx) const {
        assert(idx < size());
        if (isTransposed_) idx = toRealIndex(idx);
        return data()[idx];
    }

    Vector<T> reshape(VectorShape &shape) {
        VectorShape s(shape);
        int iuns = -1, sz = 1;
        int nr = shape.size();
        for (int i = 0; i < nr; ++i) {
            if (s[i] == -1) {
                assert(iuns == -1);
                iuns = i;
            } else {
                sz *= s[i];
            }
        }
        assert(sz == size() || (iuns != -1 && size() % sz == 0));
        if (iuns != -1) s[iuns] = size() / sz;
        return Vector<T>(*this, s);
    }

    const Vector<T> reshape(const VectorShape &shape) const {
        VectorShape s(shape);
        int iuns = -1, sz = 1;
        int nr = shape.size();
        for (int i = 0; i < nr; ++i) {
            if (s[i] == -1) {
                assert(iuns == -1);
                iuns = i;
            } else {
                sz *= s[i];
            }
        }
        assert(sz == size() || (iuns != -1 && size() % sz == 0));
        if (iuns != -1) s[iuns] = size() / sz;
        return Vector<T>(*this, s);
    }

    Vector<T> reshape(const std::initializer_list<int> &shape) const {
        return reshape(VectorShape(shape.begin(), shape.end()));
    }

    Vector<T> transpose() {
        return Vector<T>(*this, !isTransposed_);
    }

    const Vector<T> transpose() const {
        return Vector<T>(*const_cast<Vector<T> *>(this), !isTransposed_);
    }

    // Fisher-Yates shuffle algorithm
    void shuffle(std::vector<int> *indices=nullptr) {
        if (rank() == 0) return;
        int n = shape_[0];
        std::vector<int> v;
        std::vector<int> *pv = (indices ? indices : &v);
        int isz = pv->size();
        assert(!indices || isz == 0 || isz == n-1);
        int esz = (*this)[0].size();
        std::random_device seed;
        std::default_random_engine e(seed());
        if (isz == 0) pv->resize(n-1);
#pragma omp parallel for
        for (int i = n-1; i >= 1; --i) {
            std::uniform_int_distribution<> dist(0, i);
            int j = dist(e);
            if (isz == n-1) j = (*pv)[i];
            else (*pv)[i] = j;
            if (i == j) continue;
            for (int k = 0; k < esz; ++k) {
                std::swap((*this)[i].at(k), (*this)[j].at(k));
            }
        }
    }

    Vector<T> deepCopy() const {
        auto copied = Vector<T>(shape_);
        for (int i = 0; i < size(); ++i) {
            copied.at(i) = at(i);
        }
        return copied;
    }

    void debugPrint() const {
#define DBG(x) std::cerr << #x << " = " << (x) << " (L" << __LINE__ << ")" << std::endl;
        DBG(data_);
        DBG(view_);
        DBG(shape_);
        DBG(strides_);
        DBG(size_);
        DBG(isTransposed_);
        if (data_) DBG(*data_);
        if (view_) DBG(*view_);
    }
};

template<typename T>
Vector<T> operator+(const Vector<T> &lhs, const Vector<T> &rhs) {
    return lhs.deepCopy() += rhs;
}

template<typename T>
Vector<T> operator+(T lhs, const Vector<T> &rhs) {
    return rhs.deepCopy() += lhs;
}

template<typename T>
Vector<T> operator-(const Vector<T> &lhs, const Vector<T> &rhs) {
    return lhs.deepCopy() -= rhs;
}

template<typename T>
Vector<T> operator-(T lhs, const Vector<T> &rhs) {
    return rhs.deepCopy() -= lhs;
}

template<typename T>
Vector<T> operator*(const Vector<T> &lhs, const Vector<T> &rhs) {
    return lhs.deepCopy() *= rhs;
}

template<typename T>
Vector<T> operator*(T lhs, const Vector<T> &rhs) {
    return rhs.deepCopy() *= lhs;
}

template<typename T>
Vector<T> operator/(const Vector<T> &lhs, const Vector<T> &rhs) {
    return lhs.deepCopy() /= rhs;
}

template<typename T>
Vector<T> operator/(const Vector<T> &lhs, T rhs) {
    return lhs.deepCopy() /= rhs;
}

template<typename T>
Vector<T> operator%(const Vector<T> &lhs, const Vector<T> &rhs) {
    return lhs.deepCopy() %= rhs;
}

template<typename T>
Vector<T> operator%(T lhs, const Vector<T> &rhs) {
    return rhs.deepCopy() %= lhs;
}

template<typename T>
void printVector(std::ostream &os, const Vector<T> &v, int indent=0) {
    auto rank = v.rank();
    if (rank == 0) return;
    auto shape = v.shape();
    os << "[";
    if (rank == 1) {
        for (int i = 0; i < shape[0]-1; ++i) {
            os << v.at(i) << ",";
        }
        os << v.at(shape[0]-1);
    }else {
        for (int i = 0; i < shape[0]-1; ++i) {
            if (rank == 2 && i > 0) {
                for (int j = 0; j < indent; ++j) os << " ";
            }
            printVector(os, v[i], indent+1);
            os << "," << std::endl << " ";
        }
        for (int j = 0; j < indent; ++j) os << " ";
        printVector(os, v[shape[0]-1], indent+1);
    }
    os << "]";
}

template<typename T>
std::ostream &operator<<(std::ostream &os, const Vector<T> &v) {
    auto shape = v.shape();
    auto strides = v.strides();
#if 1
    os << shape << std::endl;
    os << strides << std::endl;
#endif
    printVector(os, v);
    return os;
}

template<typename T>
Vector<T> matMul(const Vector<T> &lhs, const Vector<T> &rhs) {
    assert(lhs.rank() == 2 && rhs.rank() == 2);
    auto ls = lhs.shape();
    auto rs = rhs.shape();
    assert(ls[1] == rs[0]);
    Vector<T> ret({ls[0], rs[1]}, VectorInit::Zero);
#pragma omp parallel for
    for (int i = 0; i < ls[0]; ++i) {
        for (int k = 0; k < ls[1]; ++k) {
            for (int j = 0; j < rs[1]; ++j) {
                ret.at({i, j}) += lhs.at({i, k}) * rhs.at({k, j});
            }
        }
    }
    return ret;
}

template<>
Vector<float> matMul(const Vector<float> &lhs, const Vector<float> &rhs) {
    assert(lhs.rank() == 2 && rhs.rank() == 2);
    auto ls = lhs.shape();
    auto rs = rhs.shape();
    int lrow = ls[0], lcol = ls[1];
    int rrow = rs[0], rcol = rs[1];
    assert(lcol == rrow);
    Vector<float> ret({lrow, rcol});
    auto ltrans = lhs.isTransposed();
    auto rtrans = rhs.isTransposed();
    cblas_sgemm(CblasRowMajor,
                (ltrans ? CblasTrans : CblasNoTrans),
                (rtrans ? CblasTrans : CblasNoTrans),
                lrow,
                rcol,
                lcol,
                1.,
                &lhs.at(0),
                (ltrans ? lrow : lcol),
                &rhs.at(0),
                (rtrans ? rrow : rcol),
                0.,
                &ret.at(0),
                rcol);
    return ret;
}

template<>
Vector<double> matMul(const Vector<double> &lhs, const Vector<double> &rhs) {
    assert(lhs.rank() == 2 && rhs.rank() == 2);
    auto ls = lhs.shape();
    auto rs = rhs.shape();
    int lrow = ls[0], lcol = ls[1];
    int rrow = rs[0], rcol = rs[1];
    assert(lcol == rrow);
    Vector<double> ret({lrow, rcol});
    auto ltrans = lhs.isTransposed();
    auto rtrans = rhs.isTransposed();
    cblas_dgemm(CblasRowMajor,
                (ltrans ? CblasTrans : CblasNoTrans),
                (rtrans ? CblasTrans : CblasNoTrans),
                lrow,
                rcol,
                lcol,
                1.,
                &lhs.at(0),
                (ltrans ? lrow : lcol),
                &rhs.at(0),
                (rtrans ? rrow : rcol),
                0.,
                &ret.at(0),
                rcol);
    return ret;
}

template<typename T>
Vector<T> matVecMul(const Vector<T> &lhs, const Vector<T> &rhs) {
    assert(lhs.rank() == 2 && rhs.rank() == 1);
    return matMul(lhs, rhs.reshape({rhs.size(), 1})).reshape({lhs.shape()[0]});
}

#include <cstdint>
using VectorI32 = Vector<std::int32_t>;
using VectorI64 = Vector<std::int64_t>;
using VectorF32 = Vector<float>;
using VectorF64 = Vector<double>;

using Real = float;
using VectorReal = Vector<Real>;

} // namespace mynn

#endif /* MYNN_VECTOR_HPP_ */
