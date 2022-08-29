import pybind11eg as eg
import numpy as np
import time

x = np.random.randn(3000, 30000)

t0 = time.time()
print(f"sum of x (calc by numpy): {x.sum()}")
t1 = time.time()
print(f"time (numpy): {t1-t0}s")
print(f"sum of x (calc by pybind11eg): {eg.sum(x)}")
t2 = time.time()
print(f"time (pybind11eg): {t2-t1}s")
s = 0.0
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        s += x[i][j]
print(f"sum of x (calc by raw python): {s}")
t3 = time.time()
print(f"time (raw python loop): {t3-t2}s")