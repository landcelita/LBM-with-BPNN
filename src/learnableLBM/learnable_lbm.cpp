#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// 座標は(h, w) hは下向き正に注意!! (速度も同様)

namespace py = pybind11;

class pyarr4d {
public:
    pyarr4d() {};
    pyarr4d(const ssize_t rows, const ssize_t cols, const ssize_t forbidden_rows, const ssize_t forbidden_cols, const double init) {
        shape = std::vector<ssize_t>{ rows, cols, 3, 3 };
        forbidden_at = std::vector<ssize_t>{ forbidden_rows, forbidden_cols };
        
        arr_data = new double[rows * cols * 3 * 3];
        for(int i = 0; i < rows * cols * 3 * 3; i++) {
            arr_data[i] = init;
        }
        arr = py::array_t<double>( shape, arr_data );
    }

    auto& mutable_at(const int h, const int w, const int dh, const int dw) {
        // 範囲内チェック
        if (h < forbidden_at[0] || shape[0] - forbidden_at[0] <= h || w < forbidden_at[1] || shape[1] - forbidden_at[1] <= w) {
            throw py::index_error();
        }

        return arr.mutable_at(h, w, dh+1, dw+1);
    }

    const auto at(const int h, const int w, const int dh, const int dw) const {
        // 範囲内チェック
        if (h < forbidden_at[0] || shape[0] - forbidden_at[0] <= h || w < forbidden_at[1] || shape[1] - forbidden_at[1] <= w) {
            throw py::index_error();
        }

        return arr.at(h, w, dh+1, dw+1);
    }

    const auto get_forbidden_at() const {
        return forbidden_at;
    }

    auto get_array_t_data() const {
        return arr;
    }

private:
    std::vector<ssize_t> shape{0, 0, 0, 0};
    std::vector<ssize_t> forbidden_at{0, 0}; // アクセスしてはいけない領域 [1, 2]なら上下1マス左右2マスアクセスできない
    py::array_t<double> arr = py::array_t<double>({0, 0, 0, 0}, nullptr);
    double *arr_data = nullptr;
};


class pyarr2d {
public:
    pyarr2d(const ssize_t rows, const ssize_t cols, const ssize_t forbidden_rows, const ssize_t forbidden_cols, const double init) {
        shape = std::vector<ssize_t>{ rows, cols };
        forbidden_at = std::vector<ssize_t>{ forbidden_rows, forbidden_cols };
        
        arr_data = new double[rows * cols];
        for(int i = 0; i < rows * cols; i++) {
            arr_data[i] = init;
        }
        arr = py::array_t<double>( shape, arr_data );
    }

    auto& mutable_at(const int h, const int w) {
        // 範囲内チェック
        if (h < forbidden_at[0] || shape[0] - forbidden_at[0] <= h || w < forbidden_at[1] || shape[1] - forbidden_at[1] <= w) {
            throw py::index_error();
        }

        return arr.mutable_at(h, w);
    }

    auto at(const int h, const int w) const {
        // 範囲内チェック
        if (h < forbidden_at[0] || shape[0] - forbidden_at[0] <= h || w < forbidden_at[1] || shape[1] - forbidden_at[1] <= w) {
            throw py::index_error();
        }

        return arr.at(h, w);
    }

    const auto get_forbidden_at() const {
        return forbidden_at;
    }

    auto get_array_t_data() const {
        return arr;
    }

private:
    std::vector<ssize_t> shape;
    std::vector<ssize_t> forbidden_at; // アクセスしてはいけない領域 [1, 2]なら上下1マス左右2マスアクセスできない
    py::array_t<double> arr;
    double *arr_data;
};


class LearnableLBM {
public:
    LearnableLBM(const ssize_t rows, const ssize_t cols) {
        shape = std::vector<ssize_t>{ rows, cols };

        w0_1 = pyarr4d( rows, cols, 1, 1, 0.0 );
        w1_1 = pyarr4d( rows, cols, 1, 1, 1.0 );
        w1_2 = pyarr4d( rows, cols, 1, 1, 3.0 );
        w2_2 = pyarr4d( rows, cols, 1, 1, 0.0 );
        w3_2 = pyarr4d( rows, cols, 1, 1, 4.5 );
        w4_2 = pyarr4d( rows, cols, 1, 1, -1.5 );
        w0_3 = pyarr4d( rows, cols, 2, 2, 0.0 );
        w1_3 = pyarr4d( rows, cols, 2, 2, 1.0 );
    }

    void train(const py::array_t<double> uvert_in, const py::array_t<double> uhori_in, const py::array_t<double> rho_in, const py::array_t<double> uvert_ans, const py::array_t<double> uhori_ans);
    py::array_t<double> test(const py::array_t<double> uvert_in, const py::array_t<double> uhori_in, const py::array_t<double> rho_in);

private:
    std::vector<ssize_t> shape; // 480 x 505
    pyarr4d w0_1, w1_1, w1_2, w2_2, w3_2, w4_2, w0_3, w1_3; // w(上付き添字)_(下付き添字)に注意! うまい記法が思いつかん…
    void stream1(const pyarr4d& f_0, pyarr4d& f_1, pyarr2d& u_1vert, pyarr2d& u_1hori, pyarr2d& rho_1);
    void collide2(const pyarr4d& f_1, const pyarr2d& u_1vert, const pyarr2d& u_1hori, const pyarr2d& rho_1, pyarr4d& f_2eq, pyarr4d& f_2);
    void stream3(const pyarr4d& f_2, pyarr4d& f_3, pyarr2d& u_3vert, pyarr2d& u_3hori, pyarr2d& rho_3);
    void calc_rho_and_u(const pyarr4d& f, pyarr2d& rho, pyarr2d& u_vert, pyarr2d& u_hori);
};

void LearnableLBM::stream1(const pyarr4d& f_0, pyarr4d& f_1, pyarr2d& u_1vert, pyarr2d& u_1hori, pyarr2d& rho_1) {
    for(int h = f_1.get_forbidden_at()[0]; h < shape[0] - f_1.get_forbidden_at()[0]; h++) {
        for(int w = f_1.get_forbidden_at()[1]; w < shape[1] - f_1.get_forbidden_at()[1]; w++) { // 範囲が狭まっていることに注意
            for(int dh = -1; dh <= 1; dh++) for(int dw = -1; dw <= 1; dw++) {
                f_1.mutable_at(h, w, dh, dw) = w0_1.at(h, w, dh, dw) + w1_1.at(h, w, dh, dw) * f_0.at(h-dh, w-dw, dh, dw);
            }
        }
    }

    calc_rho_and_u(f_1, rho_1, u_1vert, u_1hori);
}

void LearnableLBM::collide2(const pyarr4d& f_1, const pyarr2d& u_1vert, const pyarr2d& u_1hori, const pyarr2d& rho_1, pyarr4d& f_2eq, pyarr4d& f_2){
    const std::vector<std::vector<double>> C = {
        {1.0/36.0, 1.0/9.0, 1.0/36.0},
        {1.0/9.0, 4.0/9.0, 1.0/9.0},
        {1.0/36.0, 1.0/9.0, 1.0/36.0}
    };

    for(int h = f_2.get_forbidden_at()[0]; h < shape[0] - f_2.get_forbidden_at()[0]; h++) {
        for(int w = f_2.get_forbidden_at()[1]; w < shape[1] - f_2.get_forbidden_at()[1]; w++) {
            double upow2 = u_1hori.at(h, w) * u_1hori.at(h, w) + u_1vert.at(h, w) * u_1vert.at(h, w);

            for(int dh = -1; dh <= 1; dh++) for(int dw = -1; dw <= 1; dw++) {
                double vdotu = dh * u_1vert.at(h, w) + dw * u_1hori.at(h, w);

                f_2eq.mutable_at(h, w, dh, dw) = C[dh+1][dw+1] * rho_1.at(h, w) * (
                    1
                    + w1_2.at(h, w, dh, dw) * vdotu
                    + w2_2.at(h, w, dh, dw) * (dh * u_1hori.at(h, w) - dw * u_1vert.at(h, w))
                    + w3_2.at(h, w, dh, dw) * vdotu * vdotu
                    + w4_2.at(h, w, dh, dw) * upow2
                );

                f_2.mutable_at(h, w, dh, dw) = 0.5 * (f_1.at(h, w, dh, dw) + f_2eq.at(h, w, dh, dw));
            }
        }
    }
}

void LearnableLBM::stream3(const pyarr4d& f_2, pyarr4d& f_3, pyarr2d& u_3vert, pyarr2d& u_3hori, pyarr2d& rho_3) {
    for(int h = f_3.get_forbidden_at()[0]; h < shape[0] - f_3.get_forbidden_at()[0]; h++) {
        for(int w = f_3.get_forbidden_at()[1]; w < shape[1] - f_3.get_forbidden_at()[1]; w++) { // 範囲が狭まっていることに注意
            for(int dh = -1; dh <= 1; dh++) for(int dw = -1; dw <= 1; dw++) {
                f_3.mutable_at(h, w, dh, dw) = w0_1.at(h, w, dh, dw) + w1_1.at(h, w, dh, dw) * f_2.at(h-dh, w-dw, dh, dw);
            }
        }
    }

    calc_rho_and_u(f_3, u_3vert, u_3hori, rho_3);
}

void LearnableLBM::calc_rho_and_u(const pyarr4d& f, pyarr2d& rho, pyarr2d& u_vert, pyarr2d& u_hori) {
    for(int h = f.get_forbidden_at()[0]; h < shape[0] - f.get_forbidden_at()[0]; h++) {
        for(int w = f.get_forbidden_at()[1]; w < shape[1] - f.get_forbidden_at()[1]; w++) {
            for(int dh = -1; dh <= 1; dh++) for(int dw = -1; dw <= 1; dw++) {
                rho.mutable_at(h, w) += f.at(h, w, dh, dw);
                u_vert.mutable_at(h, w) += f.at(h, w, dh, dw) * dh;
                u_hori.mutable_at(h, w) += f.at(h, w, dh, dw) * dw;
            }
            u_vert.mutable_at(h, w) /= rho.mutable_at(h, w);
            u_hori.mutable_at(h, w) /= rho.mutable_at(h, w);
        }
    }
}

// void LearnableLBM::train(pyarr u_vert, pyarr rho_vert, pyarr u_y) {
// }

PYBIND11_MODULE(learnableLBM, m) {
    py::class_<LearnableLBM>(m, "LearnableLBM")
        .def(py::init<const ssize_t, const ssize_t>())
        .def("train", &LearnableLBM::train)
        .def("test", &LearnableLBM::test);
}
