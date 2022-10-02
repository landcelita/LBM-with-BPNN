test:
	c++ -O3 -Wall -shared -std=c++14 -DTEST_MODE -fPIC $$(python3 -m pybind11 --includes) learnable_lbm.cpp -o learnableLBM$$(python3-config --extension-suffix)
	python learnable_lbm_test.py