#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <fmt/core.h>

namespace py = pybind11;

class Matrix{
public:
    Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols) {
        m_data = new float[rows*cols];
    }
    float *data() { return m_data; }
    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }

private:
    size_t m_rows, m_cols;
    float *m_data;
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

PYBIND11_MODULE(pybind11eg, m) {
    m.doc() = "pybind11 example module";

    py::class_<Matrix>(m, "Matrix", py::buffer_protocol())
        .def_buffer([](Matrix &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),
                sizeof(float),
                py::format_descriptor<float>::format(),
                2,
                { m.rows(), m.cols() },
                { sizeof(float) * m.cols(),
                  sizeof(float) }
            );
        });

    m.def("sum", &sum<int32_t>, "");
    m.def("sum", &sum<int64_t>, "");
    m.def("sum", &sum<double>, "");

    m.attr("the_answer") = 42;
    m.attr("what") = "World";
}