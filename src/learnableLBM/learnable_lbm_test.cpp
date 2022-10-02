#include <iostream>
#include <gtest/gtest.h>
#include "learnable_lbm.hpp"

#ifndef TEST_MODE
namespace {
    TEST(Error, Error) {
        FAIL() << "You should compile with -DTEST_MODE when test.";
    }
}
#else
namespace{
    TEST(InputField, Constructor) {
        // INPUT
        std::vector<std::vector<double>> u_vert_vec = {
            {0.12, 0.8},
            {-0.07, 0.05},
        };
        std::vector<std::vector<double>> u_hori_vec = {
            {-0.03, 0.04},
            {-0.1, 0.1 },
        };
        std::vector<std::vector<double>> rho_vec = {
            {100, 110},
            {120, 90},
        };
        py::array_t<double> u_vert_arr(std::vector<ssize_t>{2, 2});
        py::array_t<double> u_hori_arr(std::vector<ssize_t>{2, 2});
        py::array_t<double> rho_arr(std::vector<ssize_t>{2, 2});
        for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) {
            u_vert_arr.mutable_at(i, j) = u_vert_vec.at(i).at(j);
            u_hori_arr.mutable_at(i, j) = u_hori_vec.at(i).at(j);
            rho_arr.mutable_at(i, j) = rho_vec.at(i).at(j);
        }
        // EXPECT
        std::vector<std::vector<std::vector<std::vector<double>>>> f_vec = {
            {
                {{2.065277778, 7.576111111, 1.745277778},
                {11.90111111, 43.42444444, 9.901111111},
                {4.245277778, 15.57611111, 3.565277778}},   // (0,0)
                {{2.116888889, 9.494222222, 2.674222222},
                {10.69688889, 48.30222222, 13.63022222},
                {3.407555556, 15.36088889, 4.316888889}}    // (0,1)
            },
            {
                {{5.392333333, 16.12933333, 2.972333333},
                {17.63533333, 52.14133333, 9.635333333},
                {3.572333333, 10.52933333, 1.992333333}},   // (1,0)
                {{1.58125, 8.425, 2.85625},
                {7.2625, 39.25, 13.2625},
                {2.10625, 11.425, 3.83125}}                 // (1,1)
            }
        };
        py::array_t<double> f_arr(std::vector<ssize_t>{2, 2, 3, 3});
        for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) for(int k = 0; k < 3; k++) for(int l = 0; l < 3; l++) {
            f_arr.mutable_at(i, j, k, l) = f_vec.at(i).at(j).at(k).at(l);
        }
        pyarr4d f_pyarr(f_arr, 0, 0);
        // EXEC
        InputField input_field(u_vert_arr, u_hori_arr, rho_arr);
        // COMPARE
        for(int h = 0; h < 2; h++) for(int w = 0; w < 2; w++) for(int dh = -1; dh <= 1; dh++) for(int dw = -1; dw <= 1; dw++) {
            EXPECT_DOUBLE_EQ(input_field.f.at(h, w, dh, dw), f_pyarr.at(h, w, dh, dw)) << "failed at (h, w, dh, dw) = (" << h << ", " << w << ", " << dh << ", " << dw << ")";
        }
    }
}
#endif