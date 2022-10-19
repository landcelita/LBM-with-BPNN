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

LearnableLBM::LearnableLBM(const ssize_t rows, const ssize_t cols):
is_dw_set(false),
streaming_weight_1_dw0(rows, cols, 1, 1, 0.0), streaming_weight_1_dw1(rows, cols, 1, 1, 0.0),
colliding_weight_dw1(rows, cols, 1, 1, 0.0), colliding_weight_dw2(rows, cols, 1, 1, 0.0), colliding_weight_dw3(rows, cols, 1, 1, 0.0), colliding_weight_dw4(rows, cols, 1, 1, 0.0),
streaming_weight_2_dw0(rows, cols, 2, 2, 0.0), streaming_weight_2_dw1(rows, cols, 2, 2, 0.0),
shape({rows, cols}),
input_field(rows, cols),
streaming_weight_1(rows, cols, 1, 1),
streamed_field_1(rows, cols, 1, 1),
colliding_weight(rows, cols, 1, 1),
collided_field(rows, cols, 1, 1),
streaming_weight_2(rows, cols, 2, 2),
streamed_field_2(rows, cols, 2, 2)
{};

// FIXME: 引数はbackwardにあわせてpyarr2dにするべきだと思う
void LearnableLBM::forward(const py::array_t<double>& u_vert_, const py::array_t<double>& u_hori_, const py::array_t<double>& rho_) {
    is_dw_set = true;

    input_field = InputField(u_vert_, u_hori_, rho_);
    streamed_field_1.stream(input_field.f, streaming_weight_1.w0, streaming_weight_1.w1);
    collided_field.collide(streamed_field_1.f, colliding_weight.w1, colliding_weight.w2, colliding_weight.w3, colliding_weight.w4);
    streamed_field_2.stream(collided_field.f, streaming_weight_2.w0, streaming_weight_2.w1);
}

void LearnableLBM::backward(const double eta, pyarr2d& u_ans_vert_, pyarr2d& u_ans_hori_){
    if(!is_dw_set) {
        py::print("first forward()");
        throw py::attribute_error();
    }

    std::tie(streaming_weight_2_dw0, streaming_weight_2_dw1) = streaming_weight_2.set_delta_and_get_dw(
        eta,
        collided_field.f,
        streamed_field_2.rho,
        streamed_field_2.u_vert,
        streamed_field_2.u_hori,
        u_ans_vert_,
        u_ans_hori_
    );
    std::tie(colliding_weight_dw1, colliding_weight_dw2, colliding_weight_dw3, colliding_weight_dw4) = colliding_weight.set_delta_and_get_dw(
        eta,
        streamed_field_1.rho,
        streamed_field_1.u_vert,
        streamed_field_1.u_hori,
        streaming_weight_2.delta,
        streaming_weight_2.w1
    );
    std::tie(streaming_weight_1_dw0, streaming_weight_1_dw1) = streaming_weight_1.set_delta_and_get_dw_2(
        eta,
        input_field.f,
        streamed_field_1.rho,
        streamed_field_1.u_vert,
        streamed_field_1.u_hori,
        collided_field.f_eq,
        colliding_weight.delta,
        colliding_weight.w1,
        colliding_weight.w2,
        colliding_weight.w3,
        colliding_weight.w4
    );

    // TODO: update

    is_dw_set = false;
};

pyarr4d::pyarr4d(const ssize_t rows, const ssize_t cols, const ssize_t forbidden_rows, const ssize_t forbidden_cols, const double init) {
    shape = std::vector<ssize_t>{ rows, cols };
    forbidden_at = std::vector<ssize_t>{ forbidden_rows, forbidden_cols };
    arr = py::array_t<double>( std::vector<ssize_t>{shape[0], shape[1], 3, 3} );
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

InputField::InputField(const ssize_t rows, const ssize_t cols):
f(rows, cols, 0, 0, 0.0), u_vert(rows, cols, 0, 0, 0.0),  u_hori(rows, cols, 0, 0, 0.0), rho(rows, cols, 0, 0, 0.0)
{}

InputField::InputField(const py::array_t<double>& u_vert_, const py::array_t<double>& u_hori_, const py::array_t<double>& rho_) : 
f(u_vert_.shape(0), u_vert_.shape(1), 0, 0, 0.0), u_vert(u_vert_, 0, 0),  u_hori(u_hori_, 0, 0), rho(rho_, 0, 0)
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

StreamedField::StreamedField(const ssize_t rows, const ssize_t cols, const ssize_t forbidden_rows, const ssize_t forbidden_cols) :
f(rows, cols, forbidden_rows, forbidden_cols, 0.0), u_vert(rows, cols, forbidden_rows, forbidden_cols, 0.0),
u_hori(rows, cols, forbidden_rows, forbidden_cols, 0.0), rho(rows, cols, forbidden_rows, forbidden_cols, 0.0) { }

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

    if(f.forbidden_at[0] != f_0.forbidden_at[0] + 1 || f.forbidden_at[1] != f_0.forbidden_at[1] + 1) {
        py::print("f.forbidden_at != f_0.forbidden_at + 1, at line", __LINE__);
        throw py::attribute_error();
    }

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

CollidedField::CollidedField(const ssize_t rows, const ssize_t cols, const ssize_t forbidden_rows, const ssize_t forbidden_cols) :
f(rows, cols, forbidden_rows, forbidden_cols, 0.0), u_vert(rows, cols, forbidden_rows, forbidden_cols, 0.0), u_hori(rows, cols, forbidden_rows, forbidden_cols, 0.0),
rho(rows, cols, forbidden_rows, forbidden_cols, 0.0), f_eq(rows, cols, forbidden_rows, forbidden_cols, 0.0) {}

void CollidedField::collide(pyarr4d f_1, pyarr4d w_1, pyarr4d w_2, pyarr4d w_3, pyarr4d w_4) {
    if(!(f.shape == f_1.shape && f_1.shape == w_1.shape && w_1.shape == w_2.shape && w_2.shape == w_3.shape && w_3.shape == w_4.shape)) {
        py::print("shapes of f, f_1, w_1, w_2, w_3, w_4 are must be the same, at line", __LINE__);
        throw py::attribute_error();
    }

    if(!(f.forbidden_at == f_1.forbidden_at && f_1.forbidden_at == w_1.forbidden_at && w_1.forbidden_at == w_2.forbidden_at 
        && w_2.forbidden_at == w_3.forbidden_at && w_3.forbidden_at == w_4.forbidden_at)) {
        py::print("forbidden_ats of f, f_1, w_1, w_2, w_3 and w_4 are must be the same, at line", __LINE__);
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

StreamingWeight::StreamingWeight(const ssize_t rows, const ssize_t cols, const ssize_t forbidden_rows, const ssize_t forbidden_cols):
w0(rows, cols, forbidden_rows, forbidden_cols, 0.0), w1(rows, cols, forbidden_rows, forbidden_cols, 1.0), delta(rows, cols, forbidden_rows, forbidden_cols, 0.0) {}

std::pair<pyarr4d, pyarr4d> StreamingWeight::set_delta_and_get_dw(double eta, pyarr4d f_prev, pyarr2d rho_next, pyarr2d u_next_vert, pyarr2d u_next_hori, pyarr2d u_ans_vert, pyarr2d u_ans_hori) {
    if(!(rho_next.shape == u_next_vert.shape &&
        u_next_vert.shape == u_next_hori.shape &&
        u_next_hori.shape == u_ans_vert.shape && 
        u_ans_vert.shape == u_ans_hori.shape &&
        f_prev.shape == rho_next.shape && 
        w0.shape == f_prev.shape)) {
        py::print("all arg's shapes are must be the same, at line", __LINE__);
        throw py::attribute_error();
    }

    if(!(rho_next.forbidden_at == u_next_vert.forbidden_at && 
        u_next_vert.forbidden_at == u_next_hori.forbidden_at && 
        u_next_hori.forbidden_at == u_ans_vert.forbidden_at && 
        u_ans_vert.forbidden_at == u_ans_hori.forbidden_at && 
        f_prev.forbidden_at[0] == rho_next.forbidden_at[0] - 1 && 
        f_prev.forbidden_at[1] == rho_next.forbidden_at[1] - 1 && 
        w0.forbidden_at == u_ans_hori.forbidden_at)) {
        py::print("arg's forbidden_at is illigal, at line", __LINE__);
        throw py::attribute_error();
    }

    pyarr4d dw0(w0.shape[0], w0.shape[1], w0.forbidden_at[0], w0.forbidden_at[1], 0.0);
    pyarr4d dw1(w0.shape[0], w0.shape[1], w0.forbidden_at[0], w0.forbidden_at[1], 0.0);

    for(int h = delta.forbidden_at[0]; h < delta.shape[0] - delta.forbidden_at[0]; h++) {
        for(int w = delta.forbidden_at[1]; w < delta.shape[1] - delta.forbidden_at[1]; w++) {
            double inv_rho = 1 / rho_next.at(h, w);
            for(int dh = -1; dh <= 1; dh++) for(int dw = -1; dw <= 1; dw++) delta.mutable_at(h, w, dh, dw) = inv_rho;
            for(int dh = -1; dh <= 1; dh++) for(int dw = -1; dw <= 1; dw++) {
                delta.mutable_at(h, w, dh, dw) *= 
                    (u_next_vert.at(h, w) - u_ans_vert.at(h, w)) * (dh - u_next_vert.at(h, w)) 
                    + (u_next_hori.at(h, w) - u_ans_hori.at(h, w)) * (dw - u_next_hori.at(h, w));
                dw0.mutable_at(h, w, dh, dw) = -eta * delta.at(h, w, dh, dw);
                dw1.mutable_at(h, w, dh, dw) = dw0.at(h, w, dh, dw) * f_prev.at(h - dh, w - dw, dh, dw);
            }
        }
    }

    return {dw0, dw1};
};

std::pair<pyarr4d, pyarr4d> StreamingWeight::set_delta_and_get_dw_2(double eta, pyarr4d f_prev, pyarr2d rho_next, pyarr2d u_next_vert, pyarr2d u_next_hori, pyarr4d f_next_next_eq, pyarr4d delta_next, pyarr4d w_next_1, pyarr4d w_next_2, pyarr4d w_next_3, pyarr4d w_next_4){
    if(!(f_prev.shape == rho_next.shape && 
        rho_next.shape == u_next_vert.shape &&
        u_next_vert.shape == u_next_hori.shape &&
        u_next_hori.shape == f_next_next_eq.shape && 
        f_next_next_eq.shape == w_next_1.shape &&
        w_next_1.shape == w_next_2.shape &&
        w_next_2.shape == w_next_3.shape &&
        w_next_3.shape == w_next_4.shape &&
        w0.shape == f_prev.shape)) {
        py::print("all arg's shapes are must be the same, at line", __LINE__);
        throw py::attribute_error();
    }

    if(!(rho_next.forbidden_at == u_next_vert.forbidden_at && 
        u_next_vert.forbidden_at == u_next_hori.forbidden_at && 
        u_next_hori.forbidden_at == f_next_next_eq.forbidden_at && 
        f_next_next_eq.forbidden_at == delta_next.forbidden_at && 
        delta_next.forbidden_at == w_next_1.forbidden_at && 
        w_next_1.forbidden_at == w_next_2.forbidden_at && 
        w_next_2.forbidden_at == w_next_3.forbidden_at && 
        w_next_3.forbidden_at == w_next_4.forbidden_at && 
        f_prev.forbidden_at[0] == rho_next.forbidden_at[0] - 1 && 
        f_prev.forbidden_at[1] == rho_next.forbidden_at[1] - 1 && 
        w0.forbidden_at == u_next_vert.forbidden_at)) {
        py::print("arg's forbidden_at is illigal, at line", __LINE__);
        throw py::attribute_error();
    }
    
    pyarr4d dw0(w0.shape[0], w0.shape[1], w0.forbidden_at[0], w0.forbidden_at[1], 0.0);
    pyarr4d dw1(w0.shape[0], w0.shape[1], w0.forbidden_at[0], w0.forbidden_at[1], 0.0);

    for(int h = delta.forbidden_at[0]; h < delta.shape[0] - delta.forbidden_at[0]; h++) {
        for(int w = delta.forbidden_at[1]; w < delta.shape[1] - delta.forbidden_at[1]; w++) {
            double sum0 = 0.0, sum1_vert = 0.0, sum1_hori = 0.0, sum2_vert = 0.0, sum2_hori = 0.0, sum3_vert = 0.0, sum3_hori = 0.0, sum4_vert = 0.0, sum4_hori = 0.0;
            for(int dh = -1; dh <= 1; dh++) for(int dw = -1; dw <= 1; dw++) {
                sum0 += delta_next.at(h, w, dh, dw) * f_next_next_eq.at(h, w, dh, dw);
            }
            sum0 /= 2.0 * rho_next.at(h, w);

            for(int dh = -1; dh <= 1; dh++) for(int dw = -1; dw <= 1; dw++) {
                sum1_vert += (dh == 0 ? 0.0 : delta_next.at(h, w, dh, dw) * C[dh+1][dw+1] * w_next_1.at(h, w, dh, dw) * dh);
                sum1_hori += (dw == 0 ? 0.0 : delta_next.at(h, w, dh, dw) * C[dh+1][dw+1] * w_next_1.at(h, w, dh, dw) * dw);
            }
            sum1_vert *= 0.5 * rho_next.at(h, w);
            sum1_hori *= 0.5 * rho_next.at(h, w);
            
            for(int dh = -1; dh <= 1; dh++) for(int dw = -1; dw <= 1; dw++) {
                sum2_vert += (dh == 0 ? 0.0 : delta_next.at(h, w, dh, dw) * C[dh+1][dw+1] * w_next_2.at(h, w, dh, dw) * dh);
                sum2_hori += (dw == 0 ? 0.0 : delta_next.at(h, w, dh, dw) * C[dh+1][dw+1] * w_next_2.at(h, w, dh, dw) * dw);
            }
            sum2_vert *= 0.5 * rho_next.at(h, w);
            sum2_hori *= 0.5 * rho_next.at(h, w);

            for(int dh = -1; dh <= 1; dh++) for(int dw = -1; dw <= 1; dw++) {
                sum3_vert += (dh == 0 ? 0.0 : delta_next.at(h, w, dh, dw) * C[dh+1][dw+1] * w_next_3.at(h, w, dh, dw) * (dh * u_next_vert.at(h, w) + dw * u_next_hori.at(h, w)) * dh);
                sum3_hori += (dw == 0 ? 0.0 : delta_next.at(h, w, dh, dw) * C[dh+1][dw+1] * w_next_3.at(h, w, dh, dw) * (dh * u_next_vert.at(h, w) + dw * u_next_hori.at(h, w)) * dw);
            }
            sum3_vert *= rho_next.at(h, w);
            sum3_hori *= rho_next.at(h, w);

            for(int dh = -1; dh <= 1; dh++) for(int dw = -1; dw <= 1; dw++) {
                sum4_vert += delta_next.at(h, w, dh, dw) * C[dh+1][dw+1] * w_next_4.at(h, w, dh, dw);
            }
            sum4_hori = sum4_vert;
            sum4_vert *= rho_next.at(h, w) * u_next_vert.at(h, w);
            sum4_hori *= rho_next.at(h, w) * u_next_hori.at(h, w);

            double sum_1_3_4_vert = sum1_vert + sum3_vert + sum4_vert;
            double sum_1_3_4_hori = sum1_hori + sum3_hori + sum4_hori;
            double rho_inv = 1 / rho_next.at(h, w);
            for(int dh = -1; dh <= 1; dh++) for(int dw = -1; dw <= 1; dw++) {
                delta.mutable_at(h, w, dh, dw) = 0.5 * delta_next.at(h, w, dh, dw) + sum0 + rho_inv * (
                    sum_1_3_4_vert * (dh - u_next_vert.at(h, w)) + sum_1_3_4_hori * (dw - u_next_hori.at(h, w))
                    + sum2_vert * (dw - u_next_hori.at(h, w)) - sum2_hori * (dh - u_next_vert.at(h, w))
                );
                dw0.mutable_at(h, w, dh, dw) = -eta * delta.at(h, w, dh, dw);
                dw1.mutable_at(h, w, dh, dw) = -eta * delta.at(h, w, dh, dw) * f_prev.at(h - dh, w - dw, dh, dw);
            }
        }
    }

    return {dw0, dw1};
}

void StreamingWeight::update(const pyarr4d& dw0, const pyarr4d& dw1) {
}

CollidingWeight::CollidingWeight(const ssize_t rows, const ssize_t cols, const ssize_t forbidden_rows, const ssize_t forbidden_cols):
w1(rows, cols, forbidden_rows, forbidden_cols, 3.0),
w2(rows, cols, forbidden_rows, forbidden_cols, 0.0),
w3(rows, cols, forbidden_rows, forbidden_cols, 4.5),
w4(rows, cols, forbidden_rows, forbidden_cols, -1.5),
delta(rows, cols, forbidden_rows, forbidden_cols, 0.0) {}

std::tuple<pyarr4d, pyarr4d, pyarr4d, pyarr4d> CollidingWeight::set_delta_and_get_dw(double eta, pyarr2d rho_prev, pyarr2d u_prev_vert, pyarr2d u_prev_hori, pyarr4d delta_next, pyarr4d w_next_1){
    if(!(
        rho_prev.shape == u_prev_vert.shape &&
        u_prev_vert.shape == u_prev_hori.shape &&
        u_prev_hori.shape == delta_next.shape &&
        delta_next.shape == w_next_1.shape &&
        w_next_1.shape == w1.shape
    )){
        py::print("all arg's shapes are must be the same, at line", __LINE__);
        throw py::attribute_error();
    }

    if(!(
        rho_prev.forbidden_at == u_prev_vert.forbidden_at &&
        u_prev_vert.forbidden_at == u_prev_hori.forbidden_at &&
        delta_next.forbidden_at[0] == u_prev_hori.forbidden_at[0] + 1 &&
        delta_next.forbidden_at[1] == u_prev_hori.forbidden_at[1] + 1 &&
        w_next_1.forbidden_at == delta_next.forbidden_at &&
        rho_prev.forbidden_at == w1.forbidden_at
    )) {
        py::print("arg's forbidden_at is illigal, at line", __LINE__);
        throw py::attribute_error();
    }

    for(int h = delta_next.forbidden_at[0]; h < delta_next.shape[0] - delta_next.forbidden_at[0]; h++) {
        for(int w = delta_next.forbidden_at[1]; w < delta_next.shape[1] - delta_next.forbidden_at[1]; w++) {
            for(int dh = -1; dh <= 1; dh++) for(int dw = -1; dw <= 1; dw++) {
                delta.mutable_at(h - dh, w - dw, dh, dw) = delta_next.at(h, w, dh, dw) * w_next_1.at(h, w, dh, dw);
            }
        }
    }

    pyarr4d dw1(w1.shape[0], w1.shape[1], w1.forbidden_at[0], w1.forbidden_at[1], 0.0);
    pyarr4d dw2(w1.shape[0], w1.shape[1], w1.forbidden_at[0], w1.forbidden_at[1], 0.0);
    pyarr4d dw3(w1.shape[0], w1.shape[1], w1.forbidden_at[0], w1.forbidden_at[1], 0.0);
    pyarr4d dw4(w1.shape[0], w1.shape[1], w1.forbidden_at[0], w1.forbidden_at[1], 0.0);

    for(int h = delta.forbidden_at[0]; h < delta.shape[0] - delta.forbidden_at[0]; h++) {
        for(int w = delta.forbidden_at[1]; w < delta.shape[1] - delta.forbidden_at[1]; w++) {
            double u2 =  u_prev_vert.at(h, w) * u_prev_vert.at(h, w) + u_prev_hori.at(h, w) * u_prev_hori.at(h, w);
            for(int dh = -1; dh <= 1; dh++) for(int dw = -1; dw <= 1; dw++) {
                double common = -0.5 * eta * C[dh+1][dw+1] * delta.at(h, w, dh, dw) * rho_prev.at(h, w);
                double vu = dh * u_prev_vert.at(h, w) + dw * u_prev_hori.at(h, w);
                dw1.mutable_at(h, w, dh, dw) = common * vu;
                dw2.mutable_at(h, w, dh, dw) = common * (dh * u_prev_hori.at(h, w) - dw * u_prev_vert.at(h, w));
                dw3.mutable_at(h, w, dh, dw) = common * vu * vu;
                dw4.mutable_at(h, w, dh, dw) = common * u2;
            }
        }
    }

    return {dw1, dw2, dw3, dw4};
}

void CollidingWeight::update(const pyarr4d& dw1, const pyarr4d& dw2, const pyarr4d& dw3, const pyarr4d& dw4) {
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
        .def(py::init<const ssize_t, const ssize_t, const ssize_t, const ssize_t>())
        .def("stream", &StreamedField::stream)
        .def_readwrite("f", &StreamedField::f)
        .def_readwrite("u_vert", &StreamedField::u_vert)
        .def_readwrite("u_hori", &StreamedField::u_hori)
        .def_readwrite("rho", &StreamedField::rho);
    
    py::class_<CollidedField>(m, "CollidedField")
        .def(py::init<const ssize_t, const ssize_t, const ssize_t, const ssize_t>())
        .def("collide", &CollidedField::collide)
        .def_readwrite("f", &CollidedField::f)
        .def_readwrite("u_vert", &CollidedField::u_vert)
        .def_readwrite("u_hori", &CollidedField::u_hori)
        .def_readwrite("rho", &CollidedField::rho)
        .def_readwrite("f_eq", &CollidedField::f_eq);

    py::class_<StreamingWeight>(m, "StreamingWeight")
        .def(py::init<const ssize_t, const ssize_t, const ssize_t, const ssize_t>())
        .def_readwrite("w0", &StreamingWeight::w0)
        .def_readwrite("w1", &StreamingWeight::w1)
        .def_readwrite("delta", &StreamingWeight::delta)
        .def("set_delta_and_get_dw", &StreamingWeight::set_delta_and_get_dw)
        .def("set_delta_and_get_dw", &StreamingWeight::set_delta_and_get_dw_2)
        .def("update", &StreamingWeight::update);

    py::class_<CollidingWeight>(m, "CollidingWeight")
        .def(py::init<const ssize_t, const ssize_t, const ssize_t, const ssize_t>())
        .def_readwrite("w1", &CollidingWeight::w1)
        .def_readwrite("w2", &CollidingWeight::w2)
        .def_readwrite("w3", &CollidingWeight::w3)
        .def_readwrite("w4", &CollidingWeight::w4)
        .def_readwrite("delta", &CollidingWeight::delta)
        .def("set_delta_and_get_dw", &CollidingWeight::set_delta_and_get_dw)
        .def("update", &CollidingWeight::update);

#endif
}