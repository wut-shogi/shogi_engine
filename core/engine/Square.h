#include <cstdint>
#include "Rules.h"

namespace shogi {
namespace engine {
enum Square : int32_t {
  A9,A8,A7,A6,A5,A4,A3,A2,A1,  //
  B9,B8,B7,B6,B5,B4,B3,B2,B1,  //
  C9,C8,C7,C6,C5,C4,C3,C2,C1,  //
  D9,D8,D7,D6,D5,D4,D3,D2,D1,  //
  E9,E8,E7,E6,E5,E4,E3,E2,E1,  //
  F9,F8,F7,F6,F5,F4,F3,F2,F1,  //
  G9,G8,G7,G6,G5,G4,G3,G2,G1,  //
  H9,H8,H7,H6,H5,H4,H3,H2,H1,  //
  I9,I8,I7,I6,I5,I4,I3,I2,I1,  //
  SQUARE_SIZE,
  N = -9,
  NE = -8,
  E = 1,
  SE = 10,
  S = 9,
  SW = 8,
  W = -1,
  NW = -10,
  WHITE_PROMOTION_START = G9,
  WHITE_PAWN_LANCE_FORECED_PROMOTION_START = I9,
  WHITE_HORSE_FORCED_PROMOTION_START = H9,
  BLACK_PROMOTION_END = C1,
  BLACK_PAWN_LANCE_FORCE_PROMOTION_END = A1,
  BLACK_HORSE_FORCED_PROMOTION_END = B1,
  WHITE_PAWN_DROP = 81,
  WHITE_LANCE_DROP,
  WHITE_KNIGHT_DROP,
  WHITE_SILVER_GENERAL_DROP,
  WHITE_GOLD_GENERAL_DROP,
  WHITE_BISHOP_DROP,
  WHITE_ROOK_DROP,
  BLACK_PAWN_DROP,
  BLACK_LANCE_DROP,
  BLACK_KNIGHT_DROP,
  BLACK_SILVER_GENERAL_DROP,
  BLACK_GOLD_GENERAL_DROP,
  BLACK_BISHOP_DROP,
  BLACK_ROOK_DROP,
  NONE,
};

__host__ __device__ inline int squareToRank(Square square) {
  return square / BOARD_DIM;
}

__host__ __device__ inline int squareToFile(Square square) {
  return square % BOARD_DIM;
}

__host__ __device__ inline Square rankFileToSquare(uint32_t rank,
                                                   uint32_t file) {
  return static_cast<Square>(rank * BOARD_DIM + file);
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
__host__ __device__ inline Region squareToRegion(Square square) {
  return (Region)(square / REGION_SIZE);
}

}  // namespace engine
}  // namespace shogi