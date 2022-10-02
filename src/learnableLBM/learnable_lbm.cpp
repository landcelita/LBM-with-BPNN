#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "learnable_lbm.hpp"

// 座標は(h, w) hは下向き正に注意!! (速度も同様)

namespace py = pybind11;
std::vector<std::vector<double>> C = {
    {1.0 / 36.0,    1.0 / 9.0,  1.0 / 36.0},
    {1.0 / 9.0,     4.0 / 9.0,  1.0 / 9.0},
    {1.0 / 36.0,    1.0 / 9.0,  1.0 / 36.0}
};

pyarr4d::pyarr4d(const ssize_t rows, const ssize_t cols, const ssize_t forbidden_rows, const ssize_t forbidden_cols, const double init) {
    shape = std::vector<ssize_t>{ rows, cols, 3, 3 };
    forbidden_at = std::vector<ssize_t>{ forbidden_rows, forbidden_cols };
    arr = py::array_t<double>( shape );
    for(int i = 0; i < rows; i++) for(int j = 0; j < cols; j++) for(int k = 0; k < 3; k++) for(int l = 0; l < 3; l++) {
        arr.mutable_at(i, j, k, l) = init;
    }
}

pyarr4d::pyarr4d(const py::array_t<double> arr_, const ssize_t forbidden_rows, const ssize_t forbidden_cols) {
    shape = std::vector<ssize_t>{ arr_.shape(0), arr_.shape(1) };
    forbidden_at = std::vector<ssize_t>{ forbidden_rows, forbidden_cols };
    arr = py::array_t<double>(arr_);
}

double& pyarr4d::mutable_at(const int h, const int w, const int dh, const int dw) {
#ifdef TEST_MODE
    // 範囲内チェック
    if (h < forbidden_at[0] || shape[0] - forbidden_at[0] <= h || w < forbidden_at[1] || shape[1] - forbidden_at[1] <= w 
    || dh < -1 || dh > 1 || dw < -1 || dw > 1) {
        py::print("index error at line", __LINE__, "in file ", __FILE__);
        throw py::index_error();
    }
#endif

    return arr.mutable_at(h, w, dh+1, dw+1);
}

const double pyarr4d::at(const int h, const int w, const int dh, const int dw) const {
#ifdef TEST_MODE        
    // 範囲内チェック
    if (h < forbidden_at[0] || shape[0] - forbidden_at[0] <= h || w < forbidden_at[1] || shape[1] - forbidden_at[1] <= w 
    || dh < -1 || dh > 1 || dw < -1 || dw > 1) {
        py::print("index error at line", __LINE__, "in file ", __FILE__);
        throw py::index_error();
    }
#endif

    return arr.at(h, w, dh+1, dw+1);
}


pyarr2d::pyarr2d(const ssize_t rows, const ssize_t cols, const ssize_t forbidden_rows, const ssize_t forbidden_cols, const double init) {
    shape = std::vector<ssize_t>{ rows, cols };
    forbidden_at = std::vector<ssize_t>{ forbidden_rows, forbidden_cols };
    arr = py::array_t<double>( shape );
}

pyarr2d::pyarr2d(const py::array_t<double> arr_, const ssize_t forbidden_rows, const ssize_t forbidden_cols) {
    shape = std::vector<ssize_t>{ arr_.shape(0), arr_.shape(1) };
    forbidden_at = std::vector<ssize_t>{ forbidden_rows, forbidden_cols };
    arr = py::array_t<double>(arr_);
}

double& pyarr2d::mutable_at(const int h, const int w) {
#ifdef TEST_MODE
    // 範囲内チェック
    if (h < forbidden_at[0] || shape[0] - forbidden_at[0] <= h || w < forbidden_at[1] || shape[1] - forbidden_at[1] <= w) {
        py::print("index error at line", __LINE__, "in file ", __FILE__);
        throw py::index_error();
    }
#endif

    return arr.mutable_at(h, w);
}

double pyarr2d::at(const int h, const int w) const {
#ifdef TEST_MODE
    // 範囲内チェック
    if (h < forbidden_at[0] || shape[0] - forbidden_at[0] <= h || w < forbidden_at[1] || shape[1] - forbidden_at[1] <= w) {
        py::print("index error at line", __LINE__, "in file ", __FILE__);
        throw py::index_error();
    }
#endif

    return arr.at(h, w);
}



InputField::InputField(const py::array_t<double> u_vert_, const py::array_t<double> u_hori_, const py::array_t<double> rho_) : 
u_vert(u_vert_, 0, 0),  u_hori(u_hori_, 0, 0), rho(rho_, 0, 0), f(u_vert_.shape(0), u_vert_.shape(1), 0, 0, 0.0)
{
    if(u_vert_.ndim() != 2 || u_hori_.ndim() != 2 || rho_.ndim() != 2) {
        py::print("ndims of u_vert, u_hori, rho are must be 2, at line", __LINE__);
        throw py::attribute_error();
    }
    if(u_vert_.shape(0) != u_hori_.shape(0) || u_hori_.shape(0) != rho_.shape(0) || u_vert_.shape(1) != u_hori_.shape(1) || u_hori_.shape(1) != rho_.shape(1)){
        py::print("shapes of u_vert, u_hori, and rho are must be the same, at line", __LINE__);
        throw py::attribute_error();
    }

    // f_eqをfとする
    for(int h = 0; h < f.shape[0]; h++) {
        for(int w = 0; w < f.shape[1]; w++) {
            double uv = u_vert.at(h, w);
            double uh = u_hori.at(h, w);
            double u2 = uv * uv + uh * uh;
            for(int dh = 1; dh <= 1; dh++) for(int dw = 1; dw <= 1; dw++) {
                double uprod = uv * dh + uh * dw;
                f.mutable_at(h, w, dh, dw) = C.at(dh+1).at(dw+1) * rho.at(h, w) * ( 1 + (3.0 + 4.5 * uprod) * uprod - 1.5 * u2 );
            }
        }
    }
};