#include "sample1.h"

int Factorial(int n) {
    int result = 1;
    for(int i = 1; i <= n; i++) {
        result *= i;
    }

    return result;
}

void WrappedInt::add(int a) {
    data += a;
}

int WrappedInt::get() {
    return data;
}

void WrappedInt::hidden_subtract(int a) {
    data -= a;
}