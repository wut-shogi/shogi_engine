#pragma once
#include <array>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include "Rules.h"

#define REGION_SIZE 27
#define EMPTY_REGION 0
#define FULL_REGION 134217727

enum BitboardType {
  BEGIN = 0,
  PAWN = 0,
  LANCE,
  KNIGHT,
  SILVER_GENERAL,
  GOLD_GENERAL,
  BISHOP,
  ROOK,
  KING,

  PROMOTED,

  ALL_WHITE,
  ALL_BLACK,

  // boards needed for vertical and diagonal sliding moves generation
  OCCUPIED_ROT90,
  OCCUPIED_ROTR45,
  OCCUPIED_ROTL45,

  END,
  SIZE = END
};

struct Bitboard {
  uint32_t bb[3];

  Bitboard() : bb{0, 0, 0} {}
  Bitboard(uint32_t region1, uint32_t region2, uint32_t region3)
      : bb{region1, region2, region3} {}
  Bitboard(std::array<short, BOARD_SIZE>& mat) {
    for (int bbIdx = 0; bbIdx < REGION_DIM; bbIdx++) {
      bb[bbIdx] = 0;
      for (int i = 0; i < REGION_SIZE; i++) {
        bb[bbIdx] += mat[bbIdx * REGION_SIZE + i] ? 1 : 0;
        if (i < REGION_SIZE - 1)
          bb[bbIdx] = bb[bbIdx] << 1;
      }
    }
  }
  Bitboard(std::array<short, BOARD_SIZE>&& mat) {
    for (int bbIdx = 0; bbIdx < REGION_DIM; bbIdx++) {
      bb[bbIdx] = 0;
      for (int i = 0; i < REGION_SIZE; i++) {
        bb[bbIdx] += mat[bbIdx * REGION_SIZE + i] ? 1 : 0;
        if (i < REGION_SIZE - 1)
          bb[bbIdx] = bb[bbIdx] << 1;
      }
    }
  }

  uint32_t& operator[](size_t i) { return bb[i]; }
  const uint32_t& operator[](size_t i) const { return bb[i]; }
  Bitboard& operator=(const Bitboard& bb) {
    Bitboard& thisBB = *this;
    thisBB[0] = bb[0];
    thisBB[1] = bb[1];
    thisBB[2] = bb[2];
    return thisBB;
  }

  Bitboard operator&(const Bitboard& BB) {
    return {bb[0] & BB[0], bb[1] & BB[1], bb[2] & BB[2]};
  }

  Bitboard operator|(const Bitboard& BB) {
    return {bb[0] | BB[0], bb[1] | BB[1], bb[2] | BB[2]};
  }

  bool GetBit(int idx) const {
    int region = idx / REGION_SIZE;
    int shift = REGION_SIZE - 1 - idx % REGION_SIZE;
    return bb[region] & (1 << shift);
  }
};

namespace Bitboards {
Bitboard FULL();
Bitboard EMPTY();
Bitboard STARTING_PAWN();
Bitboard STARTING_LANCE();
Bitboard STARTING_KNIGHT();
Bitboard STARTING_SILVER_GENERAL();
Bitboard STARTING_GOLD_GENERAL();
Bitboard STARTING_BISHOP();
Bitboard STARTING_ROOK();
Bitboard STARTING_KING();
Bitboard STARTING_PROMOTED();
Bitboard STARTING_ALL_WHITE();
Bitboard STARTING_ALL_BLACK();
}  // namespace Bitboards

inline void And_BB(Bitboard& dst, const Bitboard& src1, const Bitboard& src2) {
  dst[0] = src1[0] & src2[0];
  dst[1] = src1[1] & src2[1];
  dst[2] = src1[2] & src2[2];
}

inline void Or_BB(Bitboard dst, const Bitboard& src1, const Bitboard& src2) {
  dst[0] = src1[0] | src2[0];
  dst[1] = src1[1] | src2[1];
  dst[2] = src1[2] | src2[2];
}

inline void Not_BB(Bitboard& dst, const Bitboard& src) {
  dst[0] = ~src[0];
  dst[1] = ~src[1];
  dst[2] = ~src[2];
}

inline void Clear_BB(Bitboard& dst) {
  dst = Bitboards::EMPTY();
}

inline void Set_BB(Bitboard& dst) {
  dst = Bitboards::FULL();
}

inline void Copy_BB(Bitboard& dst, const Bitboard& src) {
  std::memcpy(dst.bb, src.bb, sizeof(dst.bb));
}

inline int Rotate90Clockwise(int idx) {
  assert(idx >= 0 && idx < BOARD_SIZE);
  static const int rot90Mapping[BOARD_SIZE] = {
      72, 63, 54, 45, 36, 27, 18, 9,  0,  //
      73, 64, 55, 46, 37, 28, 19, 10, 1,  //
      74, 65, 56, 47, 38, 29, 20, 11, 2,  //
      75, 66, 57, 48, 39, 30, 21, 12, 3,  //
      76, 67, 58, 49, 40, 31, 22, 13, 4,  //
      77, 68, 59, 50, 41, 32, 23, 14, 5,  //
      78, 69, 60, 51, 42, 33, 24, 15, 6,  //
      79, 70, 61, 52, 43, 34, 25, 16, 7,  //
      80, 71, 62, 53, 44, 35, 26, 17, 8,
  };

  return rot90Mapping[idx];
}

inline int Rotate90AntiClockwise(int idx) {
  assert(idx >= 0 && idx < BOARD_SIZE);
  static const int rot90Mapping[BOARD_SIZE] = {
      8, 17, 26, 35, 44, 53, 62, 71, 80,  //
      7, 16, 25, 34, 43, 52, 61, 70, 79,  //
      6, 15, 24, 33, 42, 51, 60, 69, 78,  //
      5, 14, 23, 32, 41, 50, 59, 68, 77,  //
      4, 13, 22, 31, 40, 49, 58, 67, 76,  //
      3, 12, 21, 30, 39, 48, 57, 66, 75,  //
      2, 11, 20, 29, 38, 47, 56, 65, 74,  //
      1, 10, 19, 28, 37, 46, 55, 64, 73,  //
      0, 9,  18, 27, 36, 45, 54, 63, 72,
  };

  return rot90Mapping[idx];
}


// Rot45 nie jest odwracalneeeee whyyyyyyy

inline int Rotate45Clockwise(int idx) {
  assert(idx >= 0 && idx <= BOARD_SIZE);
  static const int rot45Mapping[BOARD_SIZE] = {
      9,  1,  18, 10, 2,  27, 19, 11, 3,   //
      36, 28, 20, 12, 4,  45, 37, 29, 21,  //
      13, 5,  54, 46, 38, 30, 22, 14, 6,   //
      63, 55, 47, 39, 31, 23, 15, 7,  0,   //
      72, 64, 56, 48, 40, 32, 24, 16, 8,   //
      73, 65, 57, 49, 41, 33, 25, 17, 80,  //
      74, 66, 58, 50, 42, 34, 26, 75, 67,  //
      59, 51, 43, 35, 76, 68, 60, 52, 44,  //
      77, 69, 61, 53, 78, 70, 62, 79, 71,
  };

  return rot45Mapping[idx];
}

inline int Rotate45AntiClockwise(int idx) {
  assert(idx >= 0 && idx <= BOARD_SIZE);
  static const int rot45Mapping[BOARD_SIZE] = {
      7,  17, 6,  16, 26, 5,  15, 25, 35, //
      4,  14, 24, 34, 44, 3,  13, 23,  33, //
      43, 53, 2,  12, 22, 32, 42, 52,  62, //
      1,  11, 21, 31, 41, 51, 61, 71,  8,//
      0,  10, 20, 30, 40, 50, 60, 70, 80,  //
      9,  19, 29, 39, 49, 59, 69, 79, 72,//
      18,  28, 38, 48, 58, 68, 78, 27, 37, //
      47, 57, 67, 77, 36, 46, 56, 66, 76, //
      45, 55, 65, 75, 54, 64, 74, 63, 73,
  };

  return rot45Mapping[idx];
}

inline Bitboard Rotate90Clockwise(const Bitboard& bb) {
  std::array<short, BOARD_SIZE> mat;
  for (int i = 0; i < BOARD_SIZE; i++) {
    mat[i] = bb.GetBit(Rotate90Clockwise(i));
  }
  return Bitboard(mat);
}

inline Bitboard Rotate90AntiClockwise(const Bitboard& bb) {
  std::array<short, BOARD_SIZE> mat;
  for (int i = 0; i < BOARD_SIZE; i++) {
    mat[i] = bb.GetBit(Rotate90AntiClockwise(i));
  }
  return Bitboard(mat);
}

inline Bitboard Rotate45Clockwise(const Bitboard& bb) {
  std::array<short, BOARD_SIZE> mat;
  for (int i = 0; i < BOARD_SIZE; i++) {
    mat[i] = bb.GetBit(Rotate45Clockwise(i));
  }
  return Bitboard(mat);
}

inline Bitboard Rotate45AntiClockwise(const Bitboard& bb) {
  std::array<short, BOARD_SIZE> mat;
  for (int i = 0; i < BOARD_SIZE; i++) {
    mat[i] = bb.GetBit(Rotate45AntiClockwise(i));
  }
  return Bitboard(mat);
}

inline Bitboard Rotate45ClockWiseAroundPoint(const Bitboard& bb, int fieldIdx) {
  std::array<short, BOARD_SIZE> mat;
  for (int i = 0; i < BOARD_SIZE; i++) {
    mat[Rotate45AntiClockwise(i)] = bb.GetBit(i);
  }
  return Bitboard(mat);
}

inline void FillRightDiagonal(Bitboard& in, int values, int diagIdx) {

}

inline void print_BB(Bitboard& src) {
  std::bitset<32> bits;
  for (int region = 0; region < REGION_DIM; region++) {
    bits = std::bitset<32>(src[region]);
    for (int row = REGION_DIM - 1; row >= 0; row--) {
      for (int col = BOARD_DIM - 1; col >= 0; col--) {
        std::cout << (bits[row * BOARD_DIM + col] ? "1" : "0") << " ";
      }
      std::cout << std::endl;
    }
  }
}