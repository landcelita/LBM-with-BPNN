#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <fmt/core.h>

namespace py = pybind11;

class AddableMatrix {
public:
    AddableMatrix(const ssize_t rows, const ssize_t cols) {
        m_data = new double[rows * cols]();
        shape = std::vector<ssize_t>{ rows, cols };
        mat = py::array_t<double>( shape, m_data );
        std::cout << mat.mutable_data(0, 0) << '\n';
    }

    auto add(py::array_t<double> x) {
        if (x.ndim() != 2) {
            throw std::runtime_error("Input should be 2-D NumPy array");
        }

        if (x.request().shape != shape) {
            throw std::runtime_error("Input should have the same size");
        }

        for(int i = 0; i < shape[0]; i++) {
            for(int j = 0; j < shape[1]; j++) {
                *mat.mutable_data(i, j) += *x.data(i, j);
                // *mat.mutable_data(i+1, j+1)のようにすると実行時、
                // Traceback (most recent call last):
                //   File "pybind11eg/example.py", line 27, in <module>
                //     m.add(np.array([[3, 4, -2], [1, 2, 3], [0, -2, 4]]))
                // IndexError: index 3 is out of bounds for axis 1 with size 3
                // というエラーが出て怒ってくれる。mutable_dataでアクセスすれば範囲外にアクセスすることもなさそうだし、
                // おそらくメモリ管理もpybind11の方でよしなにやってくれそう。
            }
        }
    }

    auto get() {
        return mat;
    }

private:
    std::vector<ssize_t> shape;
    py::array_t<double> mat;
    double* m_data;
};

template <typename T>
T sum(py::array_t<T> x)
{
    T ret = T();
    const auto &buff_info = x.request();
    const auto &shape = buff_info.shape;
    
    for(auto i = 0; i < shape[0]; i++) {
        for(auto j = 0; j < shape[1]; j++) {
            ret += *x.data(i, j);
        }
    }

    return ret;
}

template <typename T>
T sum_np(py::array_t<T> x)
{
    T ret = T();
    const auto &buff_info = x.request();
    const auto &shape = buff_info.shape;
    auto np = py::module::import("numpy");

    ret = np.attr("sum").cast<T>();

    return ret;
}

PYBIND11_MODULE(pybind11eg, m) {
    m.doc() = "pybind11 example module";

    py::class_<AddableMatrix>(m, "AddableMatrix")
        .def(py::init<const ssize_t, const ssize_t>())
        .def("add", &AddableMatrix::add)
        .def("get", &AddableMatrix::get);

    m.def("sum", &sum<int32_t>, "");
    m.def("sum", &sum<int64_t>, "");
    m.def("sum", &sum<double>, "");

    m.def("sum_np", &sum<double>, "");

    m.attr("the_answer") = 42;
    m.attr("what") = "World";
}