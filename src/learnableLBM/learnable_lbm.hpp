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

    InputField(const ssize_t rows, const ssize_t cols);
    InputField(const py::array_t<double>& u_vert_, const py::array_t<double>& u_hori_, const py::array_t<double>& rho_);
};

struct StreamingWeight {
    pyarr4d w0, w1;
    pyarr4d delta;

    StreamingWeight(const ssize_t rows, const ssize_t cols, const ssize_t forbidden_rows, const ssize_t forbidden_cols);
    std::pair<pyarr4d, pyarr4d> set_delta_and_get_dw(
        double eta,
        const pyarr4d& f_prev,
        const pyarr2d& rho_next,
        const pyarr2d& u_next_vert,
        const pyarr2d& u_next_hori,
        const pyarr2d& u_ans_vert,
        const pyarr2d& u_ans_hori);
    std::pair<pyarr4d, pyarr4d> set_delta_and_get_dw_2(
        double eta,
        const pyarr4d& f_prev,
        const pyarr2d& rho_next,
        const pyarr2d& u_next_vert,
        const pyarr2d& u_next_hori,
        const pyarr4d& f_next_next_eq,
        const pyarr4d& delta_next,
        const pyarr4d& w_next_1,
        const pyarr4d& w_next_2, 
        const pyarr4d& w_next_3, 
        const pyarr4d& w_next_4);
    void update(const pyarr4d& dw0, const pyarr4d& dw1);
};

struct StreamedField {
    pyarr4d f;
    pyarr2d u_vert, u_hori;
    pyarr2d rho;

    StreamedField(const ssize_t rows, const ssize_t cols, const ssize_t forbidden_rows, const ssize_t forbidden_cols);
    void stream(const pyarr4d& f_0, const pyarr4d& w_0, const pyarr4d& w_1);
};

struct CollidingWeight {
    pyarr4d w1, w2, w3, w4;
    pyarr4d delta;

    CollidingWeight(const ssize_t rows, const ssize_t cols, const ssize_t forbidden_rows, const ssize_t forbidden_cols);
    std::tuple<pyarr4d, pyarr4d, pyarr4d, pyarr4d> set_delta_and_get_dw(
        double eta,
        const pyarr2d& rho_prev,
        const pyarr2d& u_prev_vert,
        const pyarr2d& u_prev_hori,
        const pyarr4d& delta_next,
        const pyarr4d& w_next_1
    );
    void update(const pyarr4d& dw1, const pyarr4d& dw2, const pyarr4d& dw3, const pyarr4d& dw4);
};

struct CollidedField {
    pyarr4d f;
    pyarr2d u_vert, u_hori;
    pyarr2d rho;
    pyarr4d f_eq;

    CollidedField(const ssize_t rows, const ssize_t cols, const ssize_t forbidden_rows, const ssize_t forbidden_cols);
    void collide(const pyarr4d& f_1, const pyarr4d& w_1, const pyarr4d& w_2, const pyarr4d& w_3, const pyarr4d& w_4);
};

struct LearnableLBM {
private:
    bool is_dw_set;
    pyarr4d streaming_weight_1_dw0, streaming_weight_1_dw1;
    pyarr4d colliding_weight_dw1, colliding_weight_dw2, colliding_weight_dw3, colliding_weight_dw4;
    pyarr4d streaming_weight_2_dw0, streaming_weight_2_dw1;

public:
    std::vector<ssize_t> shape;
    InputField input_field;
    StreamingWeight streaming_weight_1;
    StreamedField streamed_field_1;
    CollidingWeight colliding_weight;
    CollidedField collided_field;
    StreamingWeight streaming_weight_2;
    StreamedField streamed_field_2; // output_field

    LearnableLBM(const ssize_t rows, const ssize_t cols);
    std::pair<const pyarr2d&, const pyarr2d&> forward(const py::array_t<double>& u_vert_, const py::array_t<double>& u_hori_, const py::array_t<double>& rho_);
    void backward(const double eta, pyarr2d& u_ans_vert_, pyarr2d& u_ans_hori_);
};

#endif // LEARNABLE_LBM_HPP