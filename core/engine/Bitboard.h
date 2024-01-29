#pragma once
#include <array>
#include <bitset>
#include <cstring>
#include <iostream>
#include <string>
#include "Rules.h"
#include "Square.h"
namespace shogi {
namespace engine {

struct Move {
  uint16_t from : 7;
  uint16_t to : 7;
  uint16_t promotion : 1;
};

namespace BB {
enum Type {
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

  END,
  SIZE = END
};
}

enum PieceValue : int16_t {
  PAWN = 10,
  LANCE = 43,
  KNIGHT = 45,
  SILVER_GENERAL = 64,
  GOLD_GENERAL = 69,
  BISHOP = 89,
  ROOK = 104,
  IN_HAND_PAWN = 12,
  IN_HAND_LANCE = 48,
  IN_HAND_KNIGHT = 51,
  IN_HAND_SILVER_GENERAL = 72,
  IN_HAND_GOLD_GENERAL = 78,
  IN_HAND_BISHOP = 111,
  IN_HAND_ROOK = 127,
  PROMOTED_PAWN = 42,
  PROMOTED_LANCE = 63,
  PROMOTED_KNIGHT = 64,
  PROMOTED_SILVER_GENERAL = 67,
  PROMOTED_BISHOP = 115,
  PROMOTED_ROOK = 130,
  MATE = 30000
};

RUNTYPE inline uint32_t isBitSet(uint32_t region, int bit) {
  return (region & (1 << bit)) >> bit;
}

struct PieceNumber {
  uint64_t WhitePawn : 4;
  uint64_t WhiteLance : 4;
  uint64_t WhiteKnight : 4;
  uint64_t WhiteSilverGeneral : 4;
  uint64_t WhiteGoldGeneral : 4;
  uint64_t WhiteBishop : 4;
  uint64_t WhiteRook : 4;
  uint64_t BlackPawn : 4;
  uint64_t BlackLance : 4;
  uint64_t BlackKnight : 4;
  uint64_t BlackSilverGeneral : 4;
  uint64_t BlackGoldGeneral : 4;
  uint64_t BlackBishop : 4;
  uint64_t BlackRook : 4;
};
union InHandLayout {
  uint64_t value;
  PieceNumber pieceNumber;

  RUNTYPE InHandLayout() { value = 0; }
};

struct Bitboard {
  uint32_t bb[3];
  RUNTYPE Bitboard() : bb{0, 0, 0} {}
  RUNTYPE Bitboard(uint32_t region1, uint32_t region2, uint32_t region3)
      : bb{region1, region2, region3} {}
  Bitboard(std::array<bool, BOARD_SIZE>& mat) {
    for (int bbIdx = 0; bbIdx < REGION_DIM; bbIdx++) {
      bb[bbIdx] = 0;
      for (int i = 0; i < REGION_SIZE; i++) {
        bb[bbIdx] += mat[bbIdx * REGION_SIZE + i] ? 1 : 0;
        if (i < REGION_SIZE - 1)
          bb[bbIdx] = bb[bbIdx] << 1;
      }
    }
  }
  Bitboard(std::array<bool, BOARD_SIZE>&& mat) {
    for (int bbIdx = 0; bbIdx < REGION_DIM; bbIdx++) {
      bb[bbIdx] = 0;
      for (int i = 0; i < REGION_SIZE; i++) {
        bb[bbIdx] += mat[bbIdx * REGION_SIZE + i] ? 1 : 0;
        if (i < REGION_SIZE - 1)
          bb[bbIdx] = bb[bbIdx] << 1;
      }
    }
  }

  RUNTYPE Bitboard(const Square square) : bb{0, 0, 0} {
    Region region = squareToRegion(square);
    bb[region] = 1 << (REGION_SIZE - 1 - square % REGION_SIZE);
  }

  RUNTYPE uint32_t& operator[](Region region) { return bb[region]; }
  RUNTYPE const uint32_t& operator[](Region region) const { return bb[region]; }
  RUNTYPE Bitboard& operator=(const Bitboard& bb) {
    Bitboard& thisBB = *this;
    thisBB[TOP] = bb[TOP];
    thisBB[MID] = bb[MID];
    thisBB[BOTTOM] = bb[BOTTOM];
    return thisBB;
  }

  RUNTYPE Bitboard& operator&=(const Bitboard& other) {
    bb[TOP] &= other[TOP];
    bb[MID] &= other[MID];
    bb[BOTTOM] &= other[BOTTOM];
    return *this;
  }

  RUNTYPE Bitboard& operator|=(const Bitboard& other) {
    bb[TOP] |= other[TOP];
    bb[MID] |= other[MID];
    bb[BOTTOM] |= other[BOTTOM];
    return *this;
  }

  RUNTYPE operator bool() const { return bb[TOP] | bb[MID] | bb[BOTTOM]; }

  RUNTYPE bool GetBit(Square square) const {
    Region region = squareToRegion(square);
    int shift = REGION_SIZE - 1 - square % REGION_SIZE;
    return (bb[region] & (1 << shift)) != 0;
  }
};

RUNTYPE inline Bitboard operator&(const Bitboard& BB1, const Bitboard& BB2) {
  return {BB1[TOP] & BB2[TOP], BB1[MID] & BB2[MID], BB1[BOTTOM] & BB2[BOTTOM]};
}

RUNTYPE inline Bitboard operator|(const Bitboard& BB1, const Bitboard& BB2) {
  return {BB1[TOP] | BB2[TOP], BB1[MID] | BB2[MID], BB1[BOTTOM] | BB2[BOTTOM]};
}

RUNTYPE inline Bitboard operator~(const Bitboard& bb) {
  return {(~bb[TOP]) & FULL_REGION, (~bb[MID]) & FULL_REGION,
          (~bb[BOTTOM]) & FULL_REGION};
}

RUNTYPE inline void setSquare(Bitboard& bb, const Square square) {
  Region regionIdx = squareToRegion(square);
  bb[regionIdx] |= 1 << (REGION_SIZE - 1 - square % REGION_SIZE);
}

RUNTYPE inline uint32_t popcount(uint32_t value) {
#ifdef __CUDA_ARCH__
  return __popc(value);
#else
  // return __popcnt(value);
  // return std::popcount(value);
  return std::bitset<32>(value).count();
#endif
}

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

inline void print_BB(Bitboard src) {
  std::bitset<32> bits;
  for (uint32_t region = 0; region < NUMBER_OF_REGIONS; region++) {
    bits = std::bitset<32>(src[static_cast<Region>(region)]);
    for (int row = REGION_DIM - 1; row >= 0; row--) {
      for (int col = BOARD_DIM - 1; col >= 0; col--) {
        std::cout << (bits[row * BOARD_DIM + col] ? "1" : "0") << " ";
      }
      std::cout << std::endl;
    }
  }
}

static const int MultiplyDeBruijnBitPosition[32] = {
    0,  1,  28, 2,  29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4,  8,
    31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6,  11, 5,  10, 9};

inline int ffs_host(uint32_t value) {
  // Use a lookup table to find the index of the least significant set bit
  return MultiplyDeBruijnBitPosition[((uint32_t)((value & -value) *
                                                 0x077CB531U)) >>
                                     27];
}

struct BitboardIterator {
 private:
  Bitboard bitboard;
  int bitPos;
  Region currentRegion = TOP;
  uint32_t squareOffset;

 public:
  RUNTYPE void Init(const Bitboard& bb) {
    bitboard = bb;
    currentRegion = TOP;
    squareOffset = 26;
  }
  RUNTYPE bool Next() {
    while (bitboard[currentRegion] == 0) {
      if (currentRegion != BOTTOM) {
        currentRegion = static_cast<Region>(currentRegion + 1);
        squareOffset += REGION_SIZE;
      } else {
        return false;
      }
    }
#ifdef __CUDA_ARCH__
    bitPos = __ffs(bitboard[currentRegion]) - 1;
#else
    bitPos = ffs_host(bitboard[currentRegion]);
#endif
    bitboard[currentRegion] &= ~(1 << bitPos);
    return true;
  }

  RUNTYPE Square GetCurrentSquare() {
    return static_cast<Square>(squareOffset - bitPos);
  }
};

}  // namespace engine
}  // namespace shogi