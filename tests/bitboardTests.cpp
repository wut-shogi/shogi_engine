#include <gtest/gtest.h>
#include <unordered_set>
#include "engine/Bitboard.h"
using namespace shogi::engine;

TEST(Bitboards, DefaultConstructor) {
  Bitboard bitboard;

  EXPECT_EQ(bitboard[TOP], 0);
  EXPECT_EQ(bitboard[MID], 0);
  EXPECT_EQ(bitboard[BOTTOM], 0);
}

TEST(Bitboards, FromRegionsConstructor) {
  Bitboard bitboard(1, 2, 3);

  EXPECT_EQ(bitboard[TOP], 1);
  EXPECT_EQ(bitboard[MID], 2);
  EXPECT_EQ(bitboard[BOTTOM], 3);
}

TEST(Bitboards, FromBoolArrayConstructor) {
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

TEST(Bitboards, FromSquareConstructor) {
  Bitboard bitboard(B5);
  EXPECT_EQ(bitboard[TOP], 8192);
  bitboard = Bitboard(D9);
  EXPECT_EQ(bitboard[MID], 67108864);
  bitboard = Bitboard(I1);
  EXPECT_EQ(bitboard[BOTTOM], 1);
}

TEST(Bitboards, GetRegion) {
  Bitboard bitboard(1, 2, 3);
  EXPECT_EQ(bitboard[TOP], 1);
  EXPECT_EQ(bitboard[MID], 2);
  EXPECT_EQ(bitboard[BOTTOM], 3);
}

TEST(Bitboards, AssignOperator) {
  Bitboard bitboard1(1, 2, 3);
  Bitboard bitboard2 = bitboard1;
  EXPECT_EQ(bitboard2[TOP], 1);
  EXPECT_EQ(bitboard2[MID], 2);
  EXPECT_EQ(bitboard2[BOTTOM], 3);
}

TEST(Bitboards, AndAssignOperator) {
  Bitboard bitboard1(1, 4, 16);
  Bitboard bitboard2(2, 4, 20);
  bitboard1 &= bitboard2;
  EXPECT_EQ(bitboard1[TOP], 1 & 2);
  EXPECT_EQ(bitboard1[MID], 4 & 4);
  EXPECT_EQ(bitboard1[BOTTOM], 16 & 16);
}

TEST(Bitboards, OrAssignOperator) {
  Bitboard bitboard1(1, 4, 16);
  Bitboard bitboard2(2, 8, 16);
  bitboard1 |= bitboard2;
  EXPECT_EQ(bitboard1[TOP], 1 | 2);
  EXPECT_EQ(bitboard1[MID], 4 | 8);
  EXPECT_EQ(bitboard1[BOTTOM], 16 | 16);
}

TEST(Bitboards, BoolOperator) {
  Bitboard bitboard1(0, 0, 0);
  Bitboard bitboard2(0, 0, 1);
  EXPECT_TRUE(bitboard2);
  EXPECT_FALSE(bitboard1);
}

TEST(Bitboards, GetBit) {
  Bitboard bitboard(1, 4, 20);
  EXPECT_TRUE(bitboard.GetBit(C1));
  EXPECT_FALSE(bitboard.GetBit(C2));
  EXPECT_TRUE(bitboard.GetBit(I3));
}

TEST(Bitboards, AndOperator) {
  Bitboard bitboard1(1, 4, 16);
  Bitboard bitboard2(2, 4, 20);
  Bitboard result = bitboard1 & bitboard2;
  EXPECT_EQ(result[TOP], 1 & 2);
  EXPECT_EQ(result[MID], 4 & 4);
  EXPECT_EQ(result[BOTTOM], 16 & 20);
}

TEST(Bitboards, OrOperator) {
  Bitboard bitboard1(1, 4, 16);
  Bitboard bitboard2(2, 8, 16);
  Bitboard result = bitboard1 | bitboard2;
  EXPECT_EQ(result[TOP], 1 | 2);
  EXPECT_EQ(result[MID], 4 | 8);
  EXPECT_EQ(result[BOTTOM], 16 | 16);
}

TEST(Bitboards, NegationOperator) {
  Bitboard bitboard(1, 4, 16);
  Bitboard result = ~bitboard;
  EXPECT_EQ(result[TOP], ~1 & FULL_REGION);
  EXPECT_EQ(result[MID], ~4 & FULL_REGION);
  EXPECT_EQ(result[BOTTOM], ~16 & FULL_REGION);
}

TEST(Bitboards, SetSquare) {
  Bitboard bitboard;
  setSquare(bitboard, C1);
  setSquare(bitboard, D9);
  setSquare(bitboard, H5);
  EXPECT_EQ(bitboard[TOP], 1);
  EXPECT_EQ(bitboard[MID], 67108864);
  EXPECT_EQ(bitboard[BOTTOM], 8192);
}

TEST(Bitboards, FFS) {
  EXPECT_EQ(ffs_host(0), 0);
  EXPECT_EQ(ffs_host(1), 0);
  EXPECT_EQ(ffs_host(8192), 13);
  EXPECT_EQ(ffs_host(20), 2);
}

TEST(Bitboards, BitboardIterator) {
  Bitboard bitboard(67112961, 67112961, 67112961);
  BitboardIterator iterator;
  iterator.Init(bitboard);
  std::unordered_set<Square> activeSquares;
  while (iterator.Next()) {
    activeSquares.insert(iterator.GetCurrentSquare());
  }

  EXPECT_EQ(activeSquares.size(), 9);
  EXPECT_TRUE(activeSquares.find(A9) != activeSquares.end());
  EXPECT_TRUE(activeSquares.find(B4) != activeSquares.end());
  EXPECT_TRUE(activeSquares.find(C1) != activeSquares.end());
  EXPECT_TRUE(activeSquares.find(D9) != activeSquares.end());
  EXPECT_TRUE(activeSquares.find(E4) != activeSquares.end());
  EXPECT_TRUE(activeSquares.find(F1) != activeSquares.end());
  EXPECT_TRUE(activeSquares.find(G9) != activeSquares.end());
  EXPECT_TRUE(activeSquares.find(H4) != activeSquares.end());
  EXPECT_TRUE(activeSquares.find(I1) != activeSquares.end());
}