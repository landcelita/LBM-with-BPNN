#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// 座標は(h, w) hは下向き正に注意!! (速度も同様)

namespace py = pybind11;

struct pyarr4d {
    std::vector<ssize_t> shape;
    std::vector<ssize_t> forbidden_at; // アクセスしてはいけない領域 [1, 2]なら上下1マス左右2マスアクセスできない
    py::array_t<double> arr;

    pyarr4d(const ssize_t rows, const ssize_t cols, const ssize_t forbidden_rows, const ssize_t forbidden_cols, const double init) {
        shape = std::vector<ssize_t>{ rows, cols, 3, 3 };
        forbidden_at = std::vector<ssize_t>{ forbidden_rows, forbidden_cols };
        arr = py::array_t<double>( shape );
        for(int i = 0; i < rows; i++) for(int j = 0; j < cols; j++) for(int k = 0; k < 3; k++) for(int l = 0; l < 3; l++) {
            arr.mutable_at(i, j, k, l) = init;
        }
    }

    auto& mutable_at(const int h, const int w, const int dh, const int dw) {
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

    const auto at(const int h, const int w, const int dh, const int dw) const {
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
};


struct pyarr2d {
    std::vector<ssize_t> shape;
    std::vector<ssize_t> forbidden_at; // アクセスしてはいけない領域 [1, 2]なら上下1マス左右2マスアクセスできない
    py::array_t<double> arr;

    pyarr2d(const ssize_t rows, const ssize_t cols, const ssize_t forbidden_rows, const ssize_t forbidden_cols, const double init) {
        shape = std::vector<ssize_t>{ rows, cols };
        forbidden_at = std::vector<ssize_t>{ forbidden_rows, forbidden_cols };
        arr = py::array_t<double>( shape );
    }

    pyarr2d(const py::array_t<double> arr_, const ssize_t forbidden_rows, const ssize_t forbidden_cols) {
        shape = std::vector<ssize_t>{ arr_.shape(0), arr_.shape(1) };
        forbidden_at = std::vector<ssize_t>{ forbidden_rows, forbidden_cols };
        arr = py::array_t<double>(arr_);
    }

    auto& mutable_at(const int h, const int w) {
#ifdef TEST_MODE
        // 範囲内チェック
        if (h < forbidden_at[0] || shape[0] - forbidden_at[0] <= h || w < forbidden_at[1] || shape[1] - forbidden_at[1] <= w) {
            py::print("index error at line", __LINE__, "in file ", __FILE__);
            throw py::index_error();
        }
#endif

        return arr.mutable_at(h, w);
    }

    auto at(const int h, const int w) const {
#ifdef TEST_MODE
        // 範囲内チェック
        if (h < forbidden_at[0] || shape[0] - forbidden_at[0] <= h || w < forbidden_at[1] || shape[1] - forbidden_at[1] <= w) {
            py::print("index error at line", __LINE__, "in file ", __FILE__);
            throw py::index_error();
        }
#endif

        return arr.at(h, w);
    }
};

struct InputField {
    pyarr4d f;
    pyarr2d u_vert;
    pyarr2d u_hori;
    pyarr2d rho;

    InputField(const py::array_t<double> u_vert_, const py::array_t<double> u_hori_, const py::array_t<double> rho_) : 
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
                for(int dh = 1; dh <= 1; dh++) for(int dw = 1; dw <= 1; dw++) {
                    double cv = uv * dh + uh * dw;
                }
            }
        }
    }
};

struct StreamingWeight {
    pyarr4d w0, w1;
    pyarr4d delta;
};

struct StreamedField {
    pyarr4d f;
    pyarr2d u_vert, u_hori;
    pyarr2d rho;
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
};

// void LearnableLBM::train(pyarr u_vert, pyarr rho_vert, pyarr u_y) {
// }

PYBIND11_MODULE(learnableLBM, m) {
    // py::class_<LearnableLBM>(m, "LearnableLBM")
    //     .def(py::init<const ssize_t, const ssize_t>())
    //     .def("train", &LearnableLBM::train)
    //     .def("test", &LearnableLBM::test);
}
