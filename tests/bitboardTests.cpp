#include <gtest/gtest.h>
#include "engine/bitboard.h"
#include <unordered_set>
using namespace shogi::engine;

TEST(Bitboard, DefaultConstructor) {
  Bitboard bitboard;

  EXPECT_EQ(bitboard[TOP], 0);
  EXPECT_EQ(bitboard[MID], 0);
  EXPECT_EQ(bitboard[BOTTOM], 0);
}

TEST(Bitboard, FromRegionsConstructor) {
  Bitboard bitboard(1, 2, 3);

  EXPECT_EQ(bitboard[TOP], 1);
  EXPECT_EQ(bitboard[MID], 2);
  EXPECT_EQ(bitboard[BOTTOM], 3);
}

TEST(Bitboard, FromBoolArrayConstructor) {
  std::array<bool, BOARD_SIZE> bitboardRepresentation = {
      1, 0, 1, 0, 1, 0, 1, 0, 1,  //
      1, 0, 1, 0, 1, 0, 1, 0, 1,  //
      1, 0, 1, 0, 1, 0, 1, 0, 1,  //
      0, 1, 0, 1, 0, 1, 0, 1, 0,  //
      0, 1, 0, 1, 0, 1, 0, 1, 0,  //
      0, 1, 0, 1, 0, 1, 0, 1, 0,  //
      1, 1, 1, 1, 1, 1, 1, 1, 1,  //
      1, 1, 1, 1, 1, 1, 1, 1, 1,  //
      1, 1, 1, 1, 1, 1, 1, 1, 1,  //
  };

  Bitboard bitboard(bitboardRepresentation);

  EXPECT_EQ(bitboard[TOP], 89566037);
  EXPECT_EQ(bitboard[MID], 44651690);
  EXPECT_EQ(bitboard[BOTTOM], 134217727);
}

TEST(Bitboard, FromSquareConstructor) {
  Bitboard bitboard(B5);
  EXPECT_EQ(bitboard[TOP], 8192);
  bitboard = Bitboard(D9);
  EXPECT_EQ(bitboard[MID], 67108864);
  bitboard = Bitboard(I1);
  EXPECT_EQ(bitboard[BOTTOM], 1);
}

TEST(Bitboard, GetRegion) {
  Bitboard bitboard(1, 2, 3);
  EXPECT_EQ(bitboard[TOP], 1);
  EXPECT_EQ(bitboard[MID], 2);
  EXPECT_EQ(bitboard[BOTTOM], 3);
}

TEST(Bitboard, AssignOperator) {
  Bitboard bitboard1(1, 2, 3);
  Bitboard bitboard2 = bitboard1;
  EXPECT_EQ(bitboard2[TOP], 1);
  EXPECT_EQ(bitboard2[MID], 2);
  EXPECT_EQ(bitboard2[BOTTOM], 3);
}

TEST(Bitboard, AndAssignOperator) {
  Bitboard bitboard1(1, 4, 16);
  Bitboard bitboard2(2, 4, 20);
  bitboard1 &= bitboard2;
  EXPECT_EQ(bitboard1[TOP], 1 & 2);
  EXPECT_EQ(bitboard1[MID], 4 & 4);
  EXPECT_EQ(bitboard1[BOTTOM], 16 & 16);
}

TEST(Bitboard, OrAssignOperator) {
  Bitboard bitboard1(1, 4, 16);
  Bitboard bitboard2(2, 8, 16);
  bitboard1 |= bitboard2;
  EXPECT_EQ(bitboard1[TOP], 1 | 2);
  EXPECT_EQ(bitboard1[MID], 4 | 8);
  EXPECT_EQ(bitboard1[BOTTOM], 16 | 16);
}

TEST(Bitboard, BoolOperator) {
  Bitboard bitboard1(0, 0, 0);
  Bitboard bitboard2(0, 0, 1);
  EXPECT_TRUE(bitboard2);
  EXPECT_FALSE(bitboard1);
}

TEST(Bitboard, GetBit) {
  Bitboard bitboard(1, 4, 20);
  EXPECT_TRUE(bitboard.GetBit(C1));
  EXPECT_FALSE(bitboard.GetBit(C2));
  EXPECT_TRUE(bitboard.GetBit(I3));
}

TEST(Bitboard, AndOperator) {
  Bitboard bitboard1(1, 4, 16);
  Bitboard bitboard2(2, 4, 20);
  Bitboard result = bitboard1 & bitboard2;
  EXPECT_EQ(result[TOP], 1 & 2);
  EXPECT_EQ(result[MID], 4 & 4);
  EXPECT_EQ(result[BOTTOM], 16 & 20);
}

TEST(Bitboard, OrOperator) {
  Bitboard bitboard1(1, 4, 16);
  Bitboard bitboard2(2, 8, 16);
  Bitboard result = bitboard1 | bitboard2;
  EXPECT_EQ(result[TOP], 1 | 2);
  EXPECT_EQ(result[MID], 4 | 8);
  EXPECT_EQ(result[BOTTOM], 16 | 16);
}

TEST(Bitboard, NegationOperator) {
  Bitboard bitboard(1, 4, 16);
  Bitboard result = ~bitboard;
  EXPECT_EQ(result[TOP], ~1 & FULL_REGION);
  EXPECT_EQ(result[MID], ~4 & FULL_REGION);
  EXPECT_EQ(result[BOTTOM], ~16 & FULL_REGION);
}

TEST(Bitboard, SetSquare) {
  Bitboard bitboard;
  setSquare(bitboard, C1);
  setSquare(bitboard, D9);
  setSquare(bitboard, H5);
  EXPECT_EQ(bitboard[TOP], 1);
  EXPECT_EQ(bitboard[MID], 67108864);
  EXPECT_EQ(bitboard[BOTTOM], 8192);
}

TEST(Bitboard, FFS) {
  EXPECT_EQ(ffs_host(0), 0);
  EXPECT_EQ(ffs_host(1), 0);
  EXPECT_EQ(ffs_host(8192), 13);
  EXPECT_EQ(ffs_host(20), 2);
}

TEST(Bitboard, BitboardIterator) {
  Bitboard bitboard(67112961, 67112961, 67112961);
  BitboardIterator iterator;
  iterator.Init(bitboard);
  std::unordered_set<Square> activeSquares;
  while (iterator.Next()) {
    activeSquares.insert(iterator.GetCurrentSquare());
  }

  EXPECT_EQ(activeSquares.size(), 9);
  EXPECT_TRUE(activeSquares.contains(A9));
  EXPECT_TRUE(activeSquares.contains(B4));
  EXPECT_TRUE(activeSquares.contains(C1));
  EXPECT_TRUE(activeSquares.contains(D9));
  EXPECT_TRUE(activeSquares.contains(E4));
  EXPECT_TRUE(activeSquares.contains(F1));
  EXPECT_TRUE(activeSquares.contains(G9));
  EXPECT_TRUE(activeSquares.contains(H4));
  EXPECT_TRUE(activeSquares.contains(I1));
}