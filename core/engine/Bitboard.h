#pragma once
#include <array>
#include <bit>
#include <bitset>
#include <cstring>
#include <iostream>
#include "Rules.h"
#include "Square.h"
namespace shogi {
namespace engine {

namespace BB {
enum Type {
  BEGIN = 0,
  PAWN = 0,
  KNIGHT,
  SILVER_GENERAL,
  GOLD_GENERAL,
  KING,
  LANCE,
  BISHOP,
  ROOK,

  PROMOTED,

  ALL_WHITE,
  ALL_BLACK,

  END,
  SIZE = END
};
}

enum Region : uint32_t {
  TOP = 0,
  MID = 1,
  BOTTOM = 2,

  REGION_SIZE = 27,
  REGION_DIM = 3,
  NUMBER_OF_REGIONS = 3,
  EMPTY_REGION = 0,
  FULL_REGION = 134217727,
};

inline Region squareToRegion(Square square) {
  return (Region)(square / REGION_SIZE);
}

inline uint32_t isBitSet(uint32_t region, int bit) {
  return (region & (1 << bit)) >> bit;
}

struct PlayerInHandPieces {
  uint16_t Pawn : 4;
  uint16_t Lance : 2;
  uint16_t Knight : 2;
  uint16_t SilverGeneral : 2;
  uint16_t GoldGeneral : 2;
  uint16_t Bishop : 1;
  uint16_t Rook : 1;
};

union InHandPieces {
  uint32_t value;
  struct {
    PlayerInHandPieces White;
    PlayerInHandPieces Black;
  };
  InHandPieces() { value = 0; }
};

struct Bitboard {
  uint32_t bb[3];
  Bitboard() : bb{0, 0, 0} {}
  Bitboard(uint32_t region1, uint32_t region2, uint32_t region3)
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

  explicit Bitboard(const Square square) : bb{0, 0, 0} {
    Region region = squareToRegion(square);
    bb[region] = 1 << (REGION_SIZE - 1 - square % REGION_SIZE);
  }

  uint32_t& operator[](Region region) { return bb[region]; }
  const uint32_t& operator[](Region region) const { return bb[region]; }
  Bitboard& operator=(const Bitboard& bb) {
    Bitboard& thisBB = *this;
    thisBB[TOP] = bb[TOP];
    thisBB[MID] = bb[MID];
    thisBB[BOTTOM] = bb[BOTTOM];
    return thisBB;
  }

  Bitboard& operator&=(const Bitboard& other) {
    bb[TOP] &= other[TOP];
    bb[MID] &= other[MID];
    bb[BOTTOM] &= other[BOTTOM];
    return *this;
  }

  Bitboard& operator|=(const Bitboard& other) {
    bb[TOP] |= other[TOP];
    bb[MID] |= other[MID];
    bb[BOTTOM] |= other[BOTTOM];
    return *this;
  }

  operator bool() const { return bb[TOP] | bb[MID] | bb[BOTTOM]; }

  bool GetBit(Square square) const {
    Region region = squareToRegion(square);
    int shift = REGION_SIZE - 1 - square % REGION_SIZE;
    return (bb[region] & (1 << shift)) != 0;
  }

  int numberOfPieces() const {
    return std::popcount<uint32_t>(bb[TOP]) + std::popcount<uint32_t>(bb[MID]) +
           std::popcount<uint32_t>(bb[BOTTOM]);
  }
};

inline Bitboard operator&(const Bitboard& BB1, const Bitboard& BB2) {
  return {BB1[TOP] & BB2[TOP], BB1[MID] & BB2[MID], BB1[BOTTOM] & BB2[BOTTOM]};
}

inline Bitboard operator|(const Bitboard& BB1, const Bitboard& BB2) {
  return {BB1[TOP] | BB2[TOP], BB1[MID] | BB2[MID], BB1[BOTTOM] | BB2[BOTTOM]};
}

inline Bitboard operator~(const Bitboard& bb) {
  return {(~bb[TOP]) & FULL_REGION, (~bb[MID]) & FULL_REGION,
          (~bb[BOTTOM]) & FULL_REGION};
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

inline void Clear_BB(Bitboard& dst) {
  dst = Bitboards::EMPTY();
}

inline void Set_BB(Bitboard& dst) {
  dst = Bitboards::FULL();
}

inline void Copy_BB(Bitboard& dst, const Bitboard& src) {
  std::memcpy(dst.bb, src.bb, sizeof(dst.bb));
}

inline Square Rotate90Clockwise(Square square) {
  static const uint32_t rot90Mapping[BOARD_SIZE] = {
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

  return static_cast<Square>(rot90Mapping[square]);
}

inline Square Rotate90AntiClockwise(Square square) {
  static const uint32_t rot90Mapping[BOARD_SIZE] = {
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

  return static_cast<Square>(rot90Mapping[square]);
}

inline Square Rotate45Clockwise(Square square) {
  static const uint32_t rot45Mapping[BOARD_SIZE] = {
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

  return static_cast<Square>(rot45Mapping[square]);
}

inline Square Rotate45AntiClockwise(Square square) {
  static const uint32_t rot45Mapping[BOARD_SIZE] = {
      7,  17, 6,  16, 26, 5,  15, 25, 35,  //
      4,  14, 24, 34, 44, 3,  13, 23, 33,  //
      43, 53, 2,  12, 22, 32, 42, 52, 62,  //
      1,  11, 21, 31, 41, 51, 61, 71, 8,   //
      0,  10, 20, 30, 40, 50, 60, 70, 80,  //
      9,  19, 29, 39, 49, 59, 69, 79, 72,  //
      18, 28, 38, 48, 58, 68, 78, 27, 37,  //
      47, 57, 67, 77, 36, 46, 56, 66, 76,  //
      45, 55, 65, 75, 54, 64, 74, 63, 73,
  };

  return static_cast<Square>(rot45Mapping[square]);
}

inline Bitboard Rotate90Clockwise(const Bitboard& bb) {
  std::array<bool, BOARD_SIZE> mat;
  for (int sq = A9; sq < SQUARE_SIZE; sq++) {
    mat[sq] = bb.GetBit(Rotate90Clockwise(static_cast<Square>(sq)));
  }
  return Bitboard(mat);
}

inline Bitboard Rotate90AntiClockwise(const Bitboard& bb) {
  std::array<bool, BOARD_SIZE> mat;
  for (int sq = A9; sq < SQUARE_SIZE; sq++) {
    mat[sq] = bb.GetBit(Rotate90AntiClockwise(static_cast<Square>(sq)));
  }
  return Bitboard(mat);
}

inline Bitboard Rotate45Clockwise(const Bitboard& bb) {
  std::array<bool, BOARD_SIZE> mat;
  for (int sq = A9; sq < SQUARE_SIZE; sq++) {
    mat[sq] = bb.GetBit(Rotate45Clockwise(static_cast<Square>(sq)));
  }
  return Bitboard(mat);
}

inline Bitboard Rotate45AntiClockwise(const Bitboard& bb) {
  std::array<bool, BOARD_SIZE> mat;
  for (int sq = A9; sq < SQUARE_SIZE; sq++) {
    mat[sq] = bb.GetBit(Rotate45AntiClockwise(static_cast<Square>(sq)));
  }
  return Bitboard(mat);
}

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

inline bool empty(const Bitboard& bb) {
  return bb[TOP] || bb[MID] || bb[BOTTOM];
}

inline void setSquare(Bitboard& bb, const Square square) {
  Region regionIdx = squareToRegion(square);
  bb[regionIdx] |= 1 << (REGION_SIZE - 1 - square % REGION_SIZE);
}

static const int MultiplyDeBruijnBitPosition[32] = {
    0,  1,  28, 2,  29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4,  8,
    31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6,  11, 5,  10, 9};

inline int ffs_host(uint32_t value) {
  // if (value == 0) {
  //   return -1;  // Handle the case where value is zero
  // }

  // Use a lookup table to find the index of the least significant set bit
  return MultiplyDeBruijnBitPosition[((uint32_t)((value & -value) *
                                                 0x077CB531U)) >>
                                     27];
}
struct BitboardIterator {
 private:
  Bitboard bitboard;
  int bitPos;
  bool occupied;
  Region currentRegion = TOP;
  uint32_t squareOffset;

 public:
  void Init(const Bitboard& bb) {
    bitboard = bb;
    currentRegion = TOP;
    squareOffset = 26;
    occupied = false;
  }
  bool Next() {
    while (bitboard[currentRegion] == 0) {
      if (currentRegion != BOTTOM) {
        currentRegion = static_cast<Region>(currentRegion + 1);
        squareOffset += REGION_SIZE;
      } else {
        return false;
      }
    }
    bitPos = ffs_host(bitboard[currentRegion]);
    bitboard[currentRegion] &= ~(1 << bitPos);
    return true;
  }

  Square GetCurrentSquare() {
    return static_cast<Square>(squareOffset - bitPos);
  }

  bool IsCurrentSquareOccupied() { return occupied; };
};
}  // namespace engine
}  // namespace shogi