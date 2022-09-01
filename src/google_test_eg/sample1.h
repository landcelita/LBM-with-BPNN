#ifndef GTEST_SAMPLES_SAMPLE1_H_
#define GTEST_SAMPLES_SAMPLE1_H_
#include <gtest/gtest.h>

int Factorial(int n);

class WrappedInt {
    friend class WrappedIntTest;
public:
    WrappedInt() {}
    WrappedInt(int data_) : data(data_) {}

    void add(int a);
    int get();

private:
    int data = 0;
    void hidden_subtract(int a);
};

#endif
