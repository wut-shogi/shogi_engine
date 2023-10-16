#pragma once
#include <array>
#include <bitset>
#include <cstdint>
#include <cstring>
#include <cassert>
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
  OCCUPIED_ROTL45,
  OCCUPIED_ROTR45,

  END,
  SIZE = END
};

struct Bitboard {
  uint32_t bb[3];

  Bitboard() : bb{0, 0, 0} {}
  Bitboard(uint32_t region1, uint32_t region2, uint32_t region3)
      : bb{region1, region2, region3} {}
  Bitboard(std::array<bool, 81>& mat) {
    Bitboard newBB;
    for (int bbIdx = 0; bbIdx < 3; bbIdx++) {
      for (int i = 0; i <= 26; i++) {
        newBB[bbIdx] += mat[bbIdx * REGION_SIZE] ? 1 : 0;
        newBB[bbIdx] << 1;
      }
    }
  }
  Bitboard(std::array<bool, 81>&& mat) {
    Bitboard newBB;
    for (int bbIdx = 0; bbIdx < 3; bbIdx++) {
      for (int i = 0; i <= 26; i++) {
        newBB[bbIdx] += mat[bbIdx * REGION_SIZE] ? 1 : 0;
        newBB[bbIdx] << 1;
      }
    }
  }

  uint32_t& operator[](size_t i) { return bb[i]; }
  const uint32_t& operator[](size_t i) const { return bb[i]; }
  Bitboard& operator=(const Bitboard& bb) {
    Bitboard& thisBB = *this;
    Copy_BB(thisBB, bb);
    return thisBB;
  }
};

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
  dst = Bitboards::EMPTY;
}

inline void Set_BB(Bitboard& dst) {
  dst = Bitboards::FULL;
}

inline void Copy_BB(Bitboard& dst, const Bitboard& src) {
  std::memcpy(dst.bb, src.bb, sizeof(dst.bb));
}

inline int Rotate90(int idx) {
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

inline int Rotate90(int idx) {
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


namespace Bitboards {
static Bitboard FULL;
static Bitboard EMPTY;
static Bitboard STARTING_PAWN;
static Bitboard STARTING_LANCE;
static Bitboard STARTING_KNIGHT;
static Bitboard STARTING_SILVER_GENERAL;
static Bitboard STARTING_GOLD_GENERAL;
static Bitboard STARTING_BISHOP;
static Bitboard STARTING_ROOK;
static Bitboard STARTING_KING;
static Bitboard STARTING_PROMOTED;
static Bitboard STARTING_ALL_WHITE;
static Bitboard STARTING_ALL_BLACK;
}  // namespace Bitboards