#pragma once
#include "Board.h"
#include "Rules.h"

namespace shogi {
namespace engine {
__host__ __device__ inline Bitboard moveN(Bitboard bb) {
  Bitboard out;
  out[TOP] =
      ((bb[TOP] << BOARD_DIM) | (bb[MID] >> (2 * BOARD_DIM))) & FULL_REGION;
  out[MID] =
      ((bb[MID] << BOARD_DIM) | (bb[BOTTOM] >> (2 * BOARD_DIM))) & FULL_REGION;
  out[BOTTOM] = (bb[BOTTOM] << BOARD_DIM) & FULL_REGION;
  return out;
}

__host__ __device__ inline Bitboard moveNE(Bitboard bb) {
  Bitboard out;
  out[TOP] = (((bb[TOP] & NOT_RIGHT_FILE) << (BOARD_DIM - 1)) |
              ((bb[MID] & NOT_RIGHT_FILE) >> (2 * BOARD_DIM + 1))) &
             FULL_REGION;
  out[MID] = (((bb[MID] & NOT_RIGHT_FILE) << (BOARD_DIM - 1)) |
              ((bb[BOTTOM] & NOT_RIGHT_FILE) >> (2 * BOARD_DIM + 1))) &
             FULL_REGION;
  out[BOTTOM] =
      ((bb[BOTTOM] & NOT_RIGHT_FILE) << (BOARD_DIM - 1)) & FULL_REGION;
  return out;
}

__host__ __device__ inline Bitboard moveE(Bitboard bb) {
  Bitboard out;
  out[TOP] = (bb[TOP] & NOT_RIGHT_FILE) >> 1;
  out[MID] = (bb[MID] & NOT_RIGHT_FILE) >> 1;
  out[BOTTOM] = (bb[BOTTOM] & NOT_RIGHT_FILE) >> 1;
  return out;
}

__host__ __device__ inline Bitboard moveSE(Bitboard bb) {
  Bitboard out;
  out[BOTTOM] = (((bb[BOTTOM] & NOT_RIGHT_FILE) >> (BOARD_DIM + 1)) |
                 ((bb[MID] & NOT_RIGHT_FILE) << (2 * BOARD_DIM - 1))) &
                FULL_REGION;
  out[MID] = (((bb[MID] & NOT_RIGHT_FILE) >> (BOARD_DIM + 1)) |
              ((bb[TOP] & NOT_RIGHT_FILE) << (2 * BOARD_DIM - 1))) &
             FULL_REGION;
  out[TOP] = ((bb[TOP] & NOT_RIGHT_FILE) >> (BOARD_DIM + 1)) & FULL_REGION;
  return out;
}

__host__ __device__ inline Bitboard moveS(Bitboard bb) {
  Bitboard out;
  out[BOTTOM] =
      ((bb[BOTTOM] >> BOARD_DIM) | (bb[MID] << (2 * BOARD_DIM))) & FULL_REGION;
  out[MID] =
      ((bb[MID] >> BOARD_DIM) | (bb[TOP] << (2 * BOARD_DIM))) & FULL_REGION;
  out[TOP] = (bb[TOP] >> BOARD_DIM) & FULL_REGION;
  return out;
}

__host__ __device__ inline Bitboard moveSW(Bitboard bb) {
  Bitboard out;
  out[BOTTOM] = (((bb[BOTTOM] & NOT_LEFT_FILE) >> (BOARD_DIM - 1)) |
                 (bb[MID] << (2 * BOARD_DIM + 1))) &
                FULL_REGION;
  out[MID] = (((bb[MID] & NOT_LEFT_FILE) >> (BOARD_DIM - 1)) |
              (bb[TOP] << (2 * BOARD_DIM + 1))) &
             FULL_REGION;
  out[TOP] = ((bb[TOP] & NOT_LEFT_FILE) >> (BOARD_DIM - 1)) & FULL_REGION;
  return out;
}

__host__ __device__ inline Bitboard moveW(Bitboard bb) {
  Bitboard out;
  out[TOP] = (bb[TOP] & NOT_LEFT_FILE) << 1;
  out[MID] = (bb[MID] & NOT_LEFT_FILE) << 1;
  out[BOTTOM] = (bb[BOTTOM] & NOT_LEFT_FILE) << 1;
  return out;
}

__host__ __device__ inline Bitboard moveNW(Bitboard bb) {
  Bitboard out;
  out[TOP] = (((bb[TOP] & NOT_LEFT_FILE) << (BOARD_DIM + 1)) |
              ((bb[MID] & NOT_LEFT_FILE) >> (2 * BOARD_DIM - 1))) &
             FULL_REGION;
  out[MID] = (((bb[MID] & NOT_LEFT_FILE) << (BOARD_DIM + 1)) |
              ((bb[BOTTOM] & NOT_LEFT_FILE) >> (2 * BOARD_DIM - 1))) &
             FULL_REGION;
  out[BOTTOM] = ((bb[BOTTOM] & NOT_LEFT_FILE) << (BOARD_DIM + 1)) & FULL_REGION;
  return out;
}

__host__ __device__ inline Bitboard getFullFile(int fileIdx) {
  uint32_t region = FIRST_FILE >> fileIdx;
  return Bitboard(region, region, region);
}

__host__ __device__ inline uint32_t getRankBlockPattern(const Bitboard& bb,
                                                 Square square) {
  const uint32_t& region = bb[squareToRegion(square)];
  uint32_t rowsBeforeInRegion = (square / BOARD_DIM) % 3;
  uint32_t result = region << 5 << (rowsBeforeInRegion * BOARD_DIM) << 1 >> 25;
  return result;
}

__host__ __device__ inline uint32_t getFileBlockPattern(const Bitboard& occupied,
                                                 Square square) {
  int offset = squareToFile(square);
  uint32_t result = 0;
  result |= isBitSet(occupied[TOP], 17 - offset) << 6;
  result |= isBitSet(occupied[TOP], 8 - offset) << 5;
  result |= isBitSet(occupied[MID], 26 - offset) << 4;
  result |= isBitSet(occupied[MID], 17 - offset) << 3;
  result |= isBitSet(occupied[MID], 8 - offset) << 2;
  result |= isBitSet(occupied[BOTTOM], 26 - offset) << 1;
  result |= isBitSet(occupied[BOTTOM], 17 - offset);
  return result;
}

namespace CPU {
const Bitboard& getRankAttacks(const Square& square, const Bitboard& occupied);
const Bitboard& getFileAttacks(const Square& square, const Bitboard& occupied);
const Bitboard& getDiagRightAttacks(const Square& square,
                                    const Bitboard& occupied);
const Bitboard& getDiagLeftAttacks(const Square& square,
                                   const Bitboard& occupied);
const Bitboard& getRankMask(const uint32_t& rank);
const Bitboard& getFileMask(const uint32_t& file);

Bitboard* getRankAttacksPtr();
Bitboard* getFileAttacksPtr();
Bitboard* getDiagRightAttacksPtr();
Bitboard* getDiagLeftAttacksPtr();
Bitboard* getRankMaskPtr();
Bitboard* getFileMaskPtr();
}  // namespace CPU
}  // namespace engine
}  // namespace shogi