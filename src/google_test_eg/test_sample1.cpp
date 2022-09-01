#include <gtest/gtest.h>
#include "sample1.h"

namespace{
    TEST(FactorialTest, Negative){
        EXPECT_EQ(1, Factorial(-5));
        EXPECT_EQ(1, Factorial(-1));
        EXPECT_GT(Factorial(-10), 0);
    }

    TEST(FactorialTest, Zero){
        EXPECT_EQ(1, Factorial(0));
    }

    TEST(FactorialTest, Positive){
        EXPECT_EQ(1, Factorial(1));
        EXPECT_EQ(2, Factorial(2));
        EXPECT_EQ(6, Factorial(3));
        EXPECT_EQ(40320, Factorial(8));
    }

    class WrappedIntTest : public ::testing::Test
    {
    protected:
        int add(int a) { 
            x.add(a); 
            return x.data; 
        }
        int hidden_subtract(int b) { 
            x.hidden_subtract(b);
            return x.data;
        }
    private:
        WrappedInt x = WrappedInt(3);
    };

    TEST_F(WrappedIntTest, add) {
        EXPECT_EQ(add(2), 5);
        EXPECT_EQ(hidden_subtract(2), 1);
    }
}