#ifndef LEARNABLE_LBM_HPP
#define LEARNABLE_LBM_HPP
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

struct pyarr4d {
public:
    std::vector<ssize_t> shape;
    std::vector<ssize_t> forbidden_at; // アクセスしてはいけない領域 [1, 2]なら上下1マス左右2マスアクセスできない
    py::array_t<double> arr;

    pyarr4d(const ssize_t rows, const ssize_t cols, const ssize_t forbidden_rows, const ssize_t forbidden_cols, const double init);
    pyarr4d(const py::array_t<double> arr_, const ssize_t forbidden_rows, const ssize_t forbidden_cols);
    double& mutable_at(const int h, const int w, const int dh, const int dw);
    const double at(const int h, const int w, const int dh, const int dw) const;
};

struct pyarr2d {
public:
    std::vector<ssize_t> shape;
    std::vector<ssize_t> forbidden_at; // アクセスしてはいけない領域 [1, 2]なら上下1マス左右2マスアクセスできない
    py::array_t<double> arr;

    pyarr2d(const ssize_t rows, const ssize_t cols, const ssize_t forbidden_rows, const ssize_t forbidden_cols, const double init);
    pyarr2d(const py::array_t<double> arr_, const ssize_t forbidden_rows, const ssize_t forbidden_cols);

    double& mutable_at(const int h, const int w);
    double at(const int h, const int w) const;
};

struct InputField {
public:
    pyarr4d f;
    pyarr2d u_vert;
    pyarr2d u_hori;
    pyarr2d rho;

    InputField(const py::array_t<double> u_vert_, const py::array_t<double> u_hori_, const py::array_t<double> rho_);
};

struct StreamingWeight {
    pyarr4d w0, w1;
    pyarr4d delta;
};

struct StreamedField {
    pyarr4d f;
    pyarr2d u_vert, u_hori;
    pyarr2d rho;

    StreamedField(const ssize_t rows, const ssize_t cols);
    void stream(pyarr4d f_0, pyarr4d w_0, pyarr4d w_1);
};

struct CollidingWeight {
    pyarr4d w1, w2, w3, w4;
    pyarr4d delta;
};

struct CollidedField {
    pyarr4d f;
    pyarr2d u_vert, u_hori;
    pyarr2d rho;
    pyarr4d f_eq;

    CollidedField(const ssize_t rows, const ssize_t cols);
    void collide(pyarr4d f_1, pyarr4d w_1, pyarr4d w_2, pyarr4d w_3, pyarr4d w_4);
};

#endif // LEARNABLE_LBM_HPP