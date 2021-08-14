#pragma once

#include "Vec.hpp"
#include "Quaternion.hpp"

namespace CUDATracer{
namespace Math {

    //row priority
    template< typename T, unsigned row, unsigned col>
    class matrix_t {

    public:
        inline __both__ static constexpr unsigned Row() { return row; }
        inline __both__ static constexpr unsigned Column() { return col; }

        T data[row*col];

        inline __both__ T& at(unsigned r, unsigned c) { return data[r * col + c]; }
        inline __both__ T at(unsigned r, unsigned c)const { return data[r * col + c]; }

        inline __both__ matrix_t() { memset(data, 0, row*col*sizeof(T)); }
        inline __both__ matrix_t(const T& fill) { for(auto& e : data){e = fill;} }
        inline __both__ matrix_t(std::initializer_list<T> d) {
            assert(d.size() == row * col); 
            auto ptr = data;
            for (auto& e : d) { *ptr++ = e; }
        }
        //template<typename _It>
        //inline __both__ matrix_t(_It& begin, _It& end) { 
        //    assert(std::distance(begin, end) == row * col);
        //    auto ptr = data;
        //    for (auto& i = begin; i != end; ++i) { *ptr++ = *i; }
        //}

    public:
        inline __both__ matrix_t<T, col, row> Transposed()const {
            matrix_t<T, col, row> result;
            for (auto i = 0; i < row; i++) {
                for (auto j = 0; j < col; j++) {
                    result.at(j, i) = at(i, j);
                }
            }
            return result;
        }

        inline __both__ T Max() const {
            T maxVal = data[0];
            for (auto& val : data) {
                if (val > maxVal) maxVal = val;
            }
            return maxVal;
        }

        inline __both__ T Min() const {
            T minVal = data[0];
            for (auto& val : data) {
                if (val < minVal) minVal = val;
            }
            return minVal;
        }

    private:
    };

    template<typename T1, typename T2, unsigned row, unsigned col>
    inline __both__ vec_t<typename BinaryOpResultType<T1, T2>::type, col>
        MatMul(const vec_t<T1, row>& v, const matrix_t<T2, row, col>& M) {
        using dtype = typename BinaryOpResultType<T1, T2>::type;
        vec_t<dtype, col> res;

        for (unsigned j = 0; j < col; j++) {
            dtype u_j = 0;
            for (unsigned i = 0; i < row; i++) {
                u_j += v[i] * M.at(i, j);
            }
            res[j] = u_j;
        }
        return res;
    }

    template<typename T1, typename T2, unsigned row, unsigned col>
    inline __both__ vec_t<typename BinaryOpResultType<T1, T2>::type, row>
        MatMul(const matrix_t<T1, row, col>& M, const vec_t<T2, col>& v) {

        using dtype = typename BinaryOpResultType<T1, T2>::type;

        vec_t<dtype, row> res;

        for (unsigned i = 0; i < row; i++) {
            dtype u_i = 0;
            for (unsigned j = 0; j < col; j++) {
                u_i += v[j] * M.at(i, j);
            }
            res[i] = u_i;
        }
        return res;
    }

    template<typename T1, typename T2>
    inline __both__ vec_t<typename BinaryOpResultType<T1, T2>::type, 4>
        MatMul(const matrix_t<T1, 4, 4>& M, const vec_t<T2, 4>& v) {

        using dtype = typename BinaryOpResultType<T1, T2>::type;

        vec_t<dtype, 4> res;

        res.x = M.data[0] * v.x + M.data[1] * v.y + M.data[2] * v.z + M.data[3] * v.w;
        res.y = M.data[4] * v.x + M.data[5] * v.y + M.data[6] * v.z + M.data[7] * v.w;
        res.z = M.data[8] * v.x + M.data[9] * v.y + M.data[10] * v.z + M.data[11] * v.w;
        res.w = M.data[12] * v.x + M.data[13] * v.y + M.data[14] * v.z + M.data[15] * v.w;

        return res;
    }

    template<typename T1, typename T2>
    inline __both__ vec_t<typename BinaryOpResultType<T1, T2>::type, 3>
        MatMul(const matrix_t<T1, 4, 4>& M, const vec_t<T2, 3>& v, const T2& append) {

        using dtype = typename BinaryOpResultType<T1, T2>::type;

        vec_t<dtype, 3> res;

        res.x = M.data[0] * v.x + M.data[1] * v.y + M.data[2] * v.z + M.data[3] * append;
        res.y = M.data[4] * v.x + M.data[5] * v.y + M.data[6] * v.z + M.data[7] * append;
        res.z = M.data[8] * v.x + M.data[9] * v.y + M.data[10] * v.z + M.data[11] * append;

        return res;
    }

    template<typename T1, typename T2, unsigned row1, unsigned col, unsigned col2>
    inline __both__ matrix_t<typename BinaryOpResultType<T1, T2>::type, row1, col2>
        MatMul(const matrix_t<T1, row1, col>& M, const matrix_t<T2, col, col2>& N) {
        using dtype = typename BinaryOpResultType<T1, T2>::type;
        matrix_t<dtype, row1, col2> res;
        for (unsigned i = 0; i < row1; i++) {
            for (unsigned j = 0; j < col2; j++) {
                dtype u_ij = 0;
                for (unsigned t = 0; t < col; t++) {
                    u_ij += M.at(i, t) * N.at(t, j);
                }
                res.at(i, j) = u_ij;
            }
        }
        return res;
    }


    //template<typename T, typename...Ty>
    //struct MatMulRetType
    //{
    //    using subTy = typename MatMulRetType<Ty...>::type;
    //    using type = decltype(MatMul(std::declval<T>(), std::declval<subTy>()));
    //};
    //
    //template<typename T>
    //struct MatMulRetType<T>
    //{
    //    using type = T;
    //};
    //
    //template<typename T, typename...Ty>
    //typename MatMulRetType<T, Ty...>::type MatMul(const T& mat,const Ty&...mats) {
    //    return MatMul(mat, MatMul(mats...));
    //}

#define _element_op(op)                                                           \
template<typename T1, typename T2, unsigned row, unsigned col>                    \
inline __both__ matrix_t<typename BinaryOpResultType<T1, T2>::type, row, col>     \
operator##op (const matrix_t<T1, row, col>& A, const matrix_t<T2, row, col>& B) { \
    using dtype = typename BinaryOpResultType<T1, T2>::type;                      \
    matrix_t<dtype, row, col> res;                                                \
    for (unsigned i = 0; i < row; i++) {                                          \
        for (unsigned j = 0; j < col; j++) {                                      \
            res.at(i, j) = A.at(i,j) op B.at(i,j);                                \
        }                                                                         \
    }                                                                             \
    return res;                                                                   \
}                                                                                 \
                                                                                  \
template<typename T1, typename T2, unsigned row, unsigned col>                    \
inline __both__ matrix_t<typename BinaryOpResultType<T1, T2>::type, row, col>     \
    operator##op(const matrix_t<T1, row, col>& A, const vec_t<T2, row>& b) {      \
    using dtype = typename BinaryOpResultType<T1, T2>::type;                      \
    matrix_t<dtype, row, col> res;                                                \
    for (unsigned i = 0; i < row; i++) {                                          \
        for (unsigned j = 0; j < col; j++) {                                      \
            res.at(i, j) = A.at(i, j) op b[i];                            \
        }                                                                         \
    }                                                                             \
    return res;                                                                   \
}                                                                                 \
                                                                                  \
template<typename T1, typename T2, unsigned row, unsigned col>                    \
inline __both__ matrix_t<typename BinaryOpResultType<T1, T2>::type, row, col>     \
    operator##op(const vec_t<T1, col>& b, const matrix_t<T2, row, col>& A) {      \
    using dtype = typename BinaryOpResultType<T1, T2>::type;                      \
    matrix_t<dtype, row, col> res;                                                \
    for (unsigned i = 0; i < row; i++) {                                          \
        for (unsigned j = 0; j < col; j++) {                                      \
            res.at(i, j) = A.at(i, j) op b[j];                            \
        }                                                                         \
    }                                                                             \
    return res;                                                                   \
}                                                                                 \
                                                                                  \
template<typename T1, typename T2, unsigned row, unsigned col>                    \
inline __both__ matrix_t<typename BinaryOpResultType<T1, T2>::type, row, col>     \
    operator##op(const matrix_t<T1, row, col>& A, const T2& b)                    \
{                                                                                 \
    using dtype = typename BinaryOpResultType<T1, T2>::type;                      \
    matrix_t<dtype, row, col> res;                                                \
    for (unsigned i = 0; i < row; i++) {                                          \
        for (unsigned j = 0; j < col; j++) {                                      \
            res.at(i, j) = A.at(i, j) op b;                                       \
        }                                                                         \
    }                                                                             \
    return res;                                                                   \
}                                                                                 

    _element_op(+)
    _element_op(-)
    _element_op(*)
    _element_op(/ )

#undef _element_op

#define _element_assign_op(op)                                               \
template<typename T1, typename T2, unsigned row, unsigned col>               \
inline __both__ matrix_t<typename BinaryOpResultType<T1, T2>::type, row, col>\
operator##op##(matrix_t<T1, row, col>& A, const matrix_t<T2, row, col>& B) { \
    for (unsigned i = 0; i < row; i++) {                                     \
        for (unsigned j = 0; j < col; j++) {                                 \
            A.at(i, j) op B.at(i, j);                                        \
        }                                                                    \
    }                                                                        \
    return A;                                                                \
}                                                                            \
                                                                             \
template<typename T1, typename T2, unsigned row, unsigned col>               \
inline __both__ matrix_t<typename BinaryOpResultType<T1, T2>::type, row, col>\
operator##op##(matrix_t<T1, row, col>& A, const vec_t<T2, row>& b) {         \
    for (unsigned i = 0; i < row; i++) {                                     \
        for (unsigned j = 0; j < col; j++) {                                 \
            A.at(i, j) op b[i];                                      \
        }                                                                    \
    }                                                                        \
    return A;                                                                \
}                                                                            \
                                                                             \
template<typename T1, typename T2, unsigned row, unsigned col>               \
inline __both__ matrix_t<typename BinaryOpResultType<T1, T2>::type, row, col>\
operator##op##(matrix_t<T1, row, col>& A, const T2& b)                       \
{                                                                            \
    for (unsigned i = 0; i < row; i++) {                                     \
        for (unsigned j = 0; j < col; j++) {                                 \
            A.at(i, j) op b;                                                 \
        }                                                                    \
    }                                                                        \
    return A;                                                                \
}                                                                            

        _element_assign_op(*= )
        _element_assign_op(/= )
        _element_assign_op(+= )
        _element_assign_op(-= )
#undef _element_assign_op

    template<typename T, unsigned N>
    inline __both__ matrix_t<T, N, N> Identity() {
        matrix_t<T, N, N> res;
        for (unsigned i = 0; i < N; i++) res.at(i, i) = 1;
        return res;
    }


    template<typename T, unsigned N>
    inline __both__ matrix_t<T, N, N> Diag(const T& val) {
        matrix_t<T, N, N> res;
        for (unsigned i = 0; i < N; i++) res.at(i, i) = val;
        return res;
    }

    using mat3x3f = matrix_t<float, 3, 3>;
    using mat3x4f = matrix_t<float, 3, 4>;
    using mat4x4f = matrix_t<float, 4, 4>;

    template<typename T>
    matrix_t<T, 4, 4> MakeTranslate(const vec_t<T, 3>& tran) {
        return matrix_t<T, 4, 4>({
            T(1), T(0), T(0), tran.x,
            T(0), T(1), T(0), tran.y,
            T(0), T(0), T(1), tran.z,
            T(0), T(0), T(0), T(1)
        });
    }

    template<typename T>
    matrix_t<T, 4, 4> MakeScale(const vec_t<T, 3>& tran) {
        return matrix_t<T, 4, 4>({
            tran.x, T(0),   T(0),   T(0),
            T(0),   tran.y, T(0),   T(0),
            T(0),   T(0),   tran.z, T(0),
            T(0),   T(0),   T(0),   T(1)
        });
    }

    template<typename T>
    matrix_t<T, 4, 4> MakeRotation(const QuaternionT<T>& q) {
    /*
        If a quaternion is represented by qw + i qx + j qy + k qz, 
        then the equivalent matrix, to represent the same rotation, is:

        1 - 2 * qy2 - 2 * qz2       2 * qx * qy - 2 * qz * qw   2 * qx * qz + 2 * qy * qw
        2 * qx * qy + 2 * qz * qw   1 - 2 * qx2 - 2 * qz2       2 * qy * qz - 2 * qx * qw
        2 * qx * qz - 2 * qy * qw   2 * qy * qz + 2 * qx * qw   1 - 2 * qx2 - 2 * qy2
        */
        //x(v.x), y(v.y), z(v.z), w(s)
        //i,j,k,r

        matrix_t<T, 4, 4> result;
        T qxx(q.i * q.i);
        T qyy(q.j * q.j);
        T qzz(q.k * q.k);
        T qxz(q.i * q.k);
        T qxy(q.i * q.j);
        T qyz(q.j * q.k);
        T qwx(q.r * q.i);
        T qwy(q.r * q.j);
        T qwz(q.r * q.k);

        result.at(0, 0) = T(1) - T(2) * (qyy + qzz);
        result.at(1, 0) = T(2) * (qxy + qwz);
        result.at(2, 0) = T(2) * (qxz - qwy);

        result.at(0, 1) = T(2) * (qxy - qwz);
        result.at(1, 1) = T(1) - T(2) * (qxx + qzz);
        result.at(2, 1) = T(2) * (qyz + qwx);

        result.at(0, 2) = T(2) * (qxz + qwy);
        result.at(1, 2) = T(2) * (qyz - qwx);
        result.at(2, 2) = T(1) - T(2) * (qxx + qyy);

        result.at(3,3) = T(1);
        return result;
    }

}
}
