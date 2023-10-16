#pragma once
#include <gtest/gtest.h>

TEST(ExampleTest, First) {
  int sum = 0;
  for (int i = 0; i <= 10; i++) {
    sum += i;
  }

  EXPECT_EQ(sum, 55);
}