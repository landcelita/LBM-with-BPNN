# 本体ファイルとテストファイルをビルドしてテストを実行する。
test:
	g++ -shared -Wall -std=c++14 -fPIC $$(python3 -m pybind11 --includes) $$(python3-config --cflags) -DTEST_MODE learnable_lbm.cpp -o liblearnable_lbm.so
	g++ -std=c++14 learnable_lbm.hpp learnable_lbm_test.cpp liblearnable_lbm.so -I$$(python3 -m pybind11 --includes) -I./ $$(python3-config --embed --ldflags --includes) -DTEST_MODE -pthread -lgtest_main -lgtest -o test.out
	./test.out