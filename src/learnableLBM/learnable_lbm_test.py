import learnableLBM as ll
import numpy as np
import unittest

class TestInputField(unittest.TestCase):
    """test class of InputField
    """

    def test_normal(self):
        arg_u_vert = np.array([
            [0.12, 0.08],
            [-0.07, 0.05]
        ])
        arg_u_hori = np.array([
            [-0.03, 0.04],
            [-0.1, 0.1]
        ])
        arg_rho = np.array([
            [100.0, 110.0],
            [120.0, 90.0]
        ])
        expected = np.array([
            [
                [[2.065277778, 7.576111111, 1.745277778],
                [11.90111111, 43.42444444, 9.901111111],
                [4.245277778, 15.57611111, 3.565277778]],   # (0, 0)
                [[2.116888889, 9.494222222, 2.674222222],
                [10.69688889, 48.30222222, 13.63022222],
                [3.407555556, 15.36088889, 4.316888889]],   # (0, 1)
            ],
            [
                [[5.392333333, 16.12933333, 2.972333333],
                [17.63533333, 52.14133333, 9.635333333],
                [3.572333333, 10.52933333, 1.992333333]],   # (1, 0)
                [[1.58125, 8.425, 2.85625],
                [7.2625, 39.25, 13.2625],
                [2.10625, 11.425, 3.83125]],                # (1, 1)
            ]
        ])
        input_field = ll.InputField(arg_u_vert, arg_u_hori, arg_rho)
        actual = input_field.f.arr
        # np.testing.assert_array_almost_equal(expected, actual)
        self.assertTrue(np.allclose(expected, actual))

if __name__ == "__main__":
    unittest.main()
