#include <iostream>
#include <gtest/gtest.h>
#ifndef TEST_MODE
namespace {
    TEST(Error, Error) {
        FAIL() << "You should compile with -DTEST_MODE when test.";
    }
}
#else
namespace{
    
}
#endif