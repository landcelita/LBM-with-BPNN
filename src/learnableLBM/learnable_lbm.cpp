#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
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
    shape = std::vector<ssize_t>{ arr_.shape(0), arr_.shape(1), 3, 3 };
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
            for(int dh = -1; dh <= 1; dh++) for(int dw = -1; dw <= 1; dw++) {
                double uprod = uv * dh + uh * dw;
                f.mutable_at(h, w, dh, dw) = C.at(dh+1).at(dw+1) * rho.at(h, w) * ( 1 + (3.0 + 4.5 * uprod) * uprod - 1.5 * u2 );
            }
        }
    }
};

StreamedField::StreamedField(const ssize_t rows, const ssize_t cols) :
f(rows, cols, 0, 0, 0.0), u_vert(rows, cols, 0, 0, 0.0), u_hori(rows, cols, 0, 0, 0.0), rho(rows, cols, 0, 0, 0.0) { }

void StreamedField::stream(pyarr4d f_0, pyarr4d w_0, pyarr4d w_1) {
    if(!(f.shape == f_0.shape && f_0.shape == w_0.shape && w_0.shape == w_1.shape)) {
        py::print("shapes of f, f_0, w_0, and w_1 are must be the same, at line", __LINE__);
        py::print("f.shape: ", f.shape[0], f.shape[1], f.shape[2], f.shape[3]);
        py::print("f_0.shape: ", f_0.shape[0], f_0.shape[1], f_0.shape[2], f_0.shape[3]);
        py::print("w_0.shape: ", w_0.shape[0], w_0.shape[1], w_0.shape[2], w_0.shape[3]);
        py::print("w_1.shape: ", w_1.shape[0], w_1.shape[1], w_1.shape[2], w_1.shape[3]);
        throw py::attribute_error();
    }

    if(!(f_0.forbidden_at == w_0.forbidden_at && w_0.forbidden_at == w_1.forbidden_at)) {
        py::print("forbidden_ats of f_0, w_0, and w_1 are must be the same, at line", __LINE__);
        py::print("f_0.forbidden_at: ", f_0.forbidden_at[0], ", ", f_0.forbidden_at[1]);
        py::print("w_0.forbidden_at: ", w_0.forbidden_at[0], ", ", w_0.forbidden_at[1]);
        py::print("w_1.forbidden_at: ", w_1.forbidden_at[0], ", ", w_1.forbidden_at[1]);
        throw py::attribute_error();
    }

    f.forbidden_at[0] = f_0.forbidden_at[0] + 1;
    f.forbidden_at[1] = f_0.forbidden_at[1] + 1;
    rho.forbidden_at = u_vert.forbidden_at = u_hori.forbidden_at = f.forbidden_at;

    for(int h = f.forbidden_at[0]; h < f.shape[0] - f.forbidden_at[0]; h++) {
        for(int w = f.forbidden_at[1]; w < f.shape[1] - f.forbidden_at[1]; w++) {
            double rho_hw = 0.0, rho_u_vert_hw = 0.0, rho_u_hori_hw = 0.0;
            for(int dh = -1; dh <= 1; dh++) for(int dw = -1; dw <= 1; dw++) {
                f.mutable_at(h, w, dh, dw) = w_0.at(h, w, dh, dw) + w_1.at(h, w, dh, dw) * f_0.at(h - dh, w - dw, dh, dw);
                rho_hw += f.at(h, w, dh, dw);
                rho_u_vert_hw += f.at(h, w, dh, dw) * dh;
                rho_u_hori_hw += f.at(h, w, dh, dw) * dw;
            }
            rho.mutable_at(h, w) = rho_hw;
            u_vert.mutable_at(h, w) = rho_u_vert_hw / rho_hw;
            u_hori.mutable_at(h, w) = rho_u_hori_hw / rho_hw;
        }
    }
}

CollidedField::CollidedField(const ssize_t rows, const ssize_t cols) :
f(rows, cols, 0, 0, 0.0), u_vert(rows, cols, 0, 0, 0.0), u_hori(rows, cols, 0, 0, 0.0), rho(rows, cols, 0, 0, 0.0), f_eq(rows, cols, 0, 0, 0.0) {}

void CollidedField::collide(pyarr4d f_1, pyarr4d w_1, pyarr4d w_2, pyarr4d w_3, pyarr4d w_4) {
    if(!(f.shape == f_1.shape && f_1.shape == w_1.shape && w_1.shape == w_2.shape && w_2.shape == w_3.shape && w_3.shape == w_4.shape)) {
        py::print("shapes of f, f_1, w_1, w_2, w_3, w_4 are must be the same, at line", __LINE__);
        throw py::attribute_error();
    }

    if(!(f_1.forbidden_at == w_1.forbidden_at && w_1.forbidden_at == w_2.forbidden_at && w_2.forbidden_at == w_3.forbidden_at && w_3.forbidden_at == w_4.forbidden_at)) {
        py::print("forbidden_ats of f_1, w_1, w_2, w_3 and w_4 are must be the same, at line", __LINE__);
        throw py::attribute_error();
    }

    f.forbidden_at = u_vert.forbidden_at = u_hori.forbidden_at = rho.forbidden_at = f_eq.forbidden_at = f_1.forbidden_at;

    for(int h = f.forbidden_at[0]; h < f.shape[0] - f.forbidden_at[0]; h++) {
        for(int w = f.forbidden_at[1]; w < f.shape[1] - f.forbidden_at[1]; w++) {
            double rho_1_hw = 0.0, u_vert_1_hw = 0.0, u_hori_1_hw = 0.0;
            for(int dh = -1; dh <= 1; dh++) for(int dw = -1; dw <= 1; dw++) {
                rho_1_hw += f_1.at(h, w, dh, dw);
                u_vert_1_hw += f_1.at(h, w, dh, dw) * dh;
                u_hori_1_hw += f_1.at(h, w, dh, dw) * dw;
            }
            
            u_vert_1_hw /= rho_1_hw; u_hori_1_hw /= rho_1_hw;

            double rho_hw = 0.0, rho_u_vert_hw = 0.0, rho_u_hori_hw = 0.0;

            for(int dh = -1; dh <= 1; dh++) for(int dw = -1; dw <= 1; dw++) {
                double vu_1 = u_vert_1_hw * dh + u_hori_1_hw * dw;
                double vxu_1 = dh * u_hori_1_hw - dw * u_vert_1_hw;
                double uu_1 = u_vert_1_hw * u_vert_1_hw + u_hori_1_hw * u_hori_1_hw;
                f_eq.mutable_at(h, w, dh, dw) = C[dh+1][dw+1] * rho_1_hw * (
                    1.0 + w_1.at(h, w, dh, dw) * vu_1 + w_2.at(h, w, dh, dw) * vxu_1 + w_3.at(h, w, dh, dw) * vu_1 * vu_1 + w_4.at(h, w, dh, dw) * uu_1
                );
                double fhw = f.mutable_at(h, w, dh, dw) = 0.5 * (f_1.at(h, w, dh, dw) + f_eq.at(h, w, dh, dw));
                rho_hw += fhw;
                rho_u_vert_hw += fhw * dh;
                rho_u_hori_hw += fhw * dw;
            }

            rho.mutable_at(h, w) = rho_hw;
            u_vert.mutable_at(h, w) = rho_u_vert_hw / rho_hw;
            u_hori.mutable_at(h, w) = rho_u_hori_hw / rho_hw;
        }
    }
}

PYBIND11_MODULE(learnableLBM, m) {
#ifndef TEST_MODE
#else
    m.doc() = "Learnable LBM TEST Module";

    py::class_<pyarr2d>(m, "pyarr2d")
        .def(py::init<const ssize_t, const ssize_t, const ssize_t, const ssize_t, const double>())
        .def(py::init<const py::array_t<double>, const ssize_t, const ssize_t>())
        .def_readwrite("arr", &pyarr2d::arr)
        .def_readwrite("shape", &pyarr2d::shape)
        .def_readwrite("forbidden_at", &pyarr2d::forbidden_at);

    py::class_<pyarr4d>(m, "pyarr4d")
        .def(py::init<const ssize_t, const ssize_t, const ssize_t, const ssize_t, const double>())
        .def(py::init<const py::array_t<double>, const ssize_t, const ssize_t>())
        .def_readwrite("arr", &pyarr4d::arr)
        .def_readwrite("shape", &pyarr4d::shape)
        .def_readwrite("forbidden_at", &pyarr4d::forbidden_at);

    py::class_<InputField>(m, "InputField")
        .def(py::init<const py::array_t<double>, const py::array_t<double>, const py::array_t<double>>())
        .def_readwrite("u_vert", &InputField::u_vert)
        .def_readwrite("u_hori", &InputField::u_hori)
        .def_readwrite("rho", &InputField::rho)
        .def_readwrite("f", &InputField::f);

    py::class_<StreamedField>(m, "StreamedField")
        .def(py::init<const ssize_t, const ssize_t>())
        .def("stream", &StreamedField::stream)
        .def_readwrite("f", &StreamedField::f)
        .def_readwrite("u_vert", &StreamedField::u_vert)
        .def_readwrite("u_hori", &StreamedField::u_hori)
        .def_readwrite("rho", &StreamedField::rho);
    
    py::class_<CollidedField>(m, "CollidedField")
        .def(py::init<const ssize_t, const ssize_t>())
        .def("collide", &CollidedField::collide)
        .def_readwrite("f", &CollidedField::f)
        .def_readwrite("u_vert", &CollidedField::u_vert)
        .def_readwrite("u_hori", &CollidedField::u_hori)
        .def_readwrite("rho", &CollidedField::rho)
        .def_readwrite("f_eq", &CollidedField::f_eq);

#endif
}