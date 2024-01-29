#pragma once
#include "Board.h"
#include "Rules.h"

namespace shogi {
namespace engine {
RUNTYPE inline Bitboard moveN(Bitboard bb) {
  Bitboard out;
  out[TOP] =
      ((bb[TOP] << BOARD_DIM) | (bb[MID] >> (2 * BOARD_DIM))) & FULL_REGION;
  out[MID] =
      ((bb[MID] << BOARD_DIM) | (bb[BOTTOM] >> (2 * BOARD_DIM))) & FULL_REGION;
  out[BOTTOM] = (bb[BOTTOM] << BOARD_DIM) & FULL_REGION;
  return out;
}

RUNTYPE inline Bitboard moveNE(Bitboard bb) {
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

RUNTYPE inline Bitboard moveE(Bitboard bb) {
  Bitboard out;
  out[TOP] = (bb[TOP] & NOT_RIGHT_FILE) >> 1;
  out[MID] = (bb[MID] & NOT_RIGHT_FILE) >> 1;
  out[BOTTOM] = (bb[BOTTOM] & NOT_RIGHT_FILE) >> 1;
  return out;
}

RUNTYPE inline Bitboard moveSE(Bitboard bb) {
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

RUNTYPE inline Bitboard moveS(Bitboard bb) {
  Bitboard out;
  out[BOTTOM] =
      ((bb[BOTTOM] >> BOARD_DIM) | (bb[MID] << (2 * BOARD_DIM))) & FULL_REGION;
  out[MID] =
      ((bb[MID] >> BOARD_DIM) | (bb[TOP] << (2 * BOARD_DIM))) & FULL_REGION;
  out[TOP] = (bb[TOP] >> BOARD_DIM) & FULL_REGION;
  return out;
}

RUNTYPE inline Bitboard moveSW(Bitboard bb) {
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

RUNTYPE inline Bitboard moveW(Bitboard bb) {
  Bitboard out;
  out[TOP] = (bb[TOP] & NOT_LEFT_FILE) << 1;
  out[MID] = (bb[MID] & NOT_LEFT_FILE) << 1;
  out[BOTTOM] = (bb[BOTTOM] & NOT_LEFT_FILE) << 1;
  return out;
}

RUNTYPE inline Bitboard moveNW(Bitboard bb) {
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

RUNTYPE inline Bitboard getFullFile(int fileIdx) {
  uint32_t region = FIRST_FILE >> fileIdx;
  return Bitboard(region, region, region);
}

RUNTYPE inline void makeMove(Board& board, Move move) {
  uint64_t one = 1;
  Region toRegionIdx = squareToRegion(static_cast<Square>(move.to));
  uint32_t toRegion = 1 << (REGION_SIZE - 1 - move.to % REGION_SIZE);
  uint32_t inHandOffsetForColor = 0;
  bool captured = false;
  if (move.from < SQUARE_SIZE) {
    Region fromRegionIdx = squareToRegion(static_cast<Square>(move.from));
    uint32_t fromRegion = 1 << (REGION_SIZE - 1 - move.from % REGION_SIZE);
    // Check if white was captured
    if (board[BB::Type::ALL_WHITE][toRegionIdx] & toRegion) {
      inHandOffsetForColor = 7;
      captured = true;
    } else if (board[BB::Type::ALL_BLACK][toRegionIdx] & toRegion) {
      captured = true;
    }
    if (captured) {
      for (int i = 0; i < BB::Type::SIZE; i++) {
        if (board[static_cast<BB::Type>(i)][toRegionIdx] & toRegion) {
          board[static_cast<BB::Type>(i)][toRegionIdx] &= ~toRegion;
          if (i < BB::Type::KING) {
            uint64_t addedValue = one << (4 * (inHandOffsetForColor + i));
            board.inHand.value += addedValue;
          }
        }
      }
    }
    for (int i = 0; i < BB::Type::SIZE; i++) {
      if (board[static_cast<BB::Type>(i)][fromRegionIdx] & fromRegion) {
        board[static_cast<BB::Type>(i)][fromRegionIdx] &= ~fromRegion;
        board[static_cast<BB::Type>(i)][toRegionIdx] |= toRegion;
      }
    }

    if (move.promotion) {
      board[BB::Type::PROMOTED][toRegionIdx] |= toRegion;
    }
  } else {
    int offset = move.from - WHITE_PAWN_DROP;
    uint64_t addedValue = one << (4 * offset);
    board.inHand.value -= addedValue;
    board[static_cast<BB::Type>(offset % 7)][toRegionIdx] |= toRegion;
    board[static_cast<BB::Type>(offset / 7 + BB::Type::ALL_WHITE)][toRegionIdx] |= toRegion;
  }
}
}  // namespace engine
}  // namespace shogi