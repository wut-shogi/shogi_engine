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

union MoveInfo {
  uint16_t value;
  struct {
    uint16_t fromType : 3;
    uint16_t fromPromotion : 1;
    uint16_t fromColor : 1;
    uint16_t toType : 3;
    uint16_t toPromotion : 1;
    uint16_t toColor : 1;
    uint16_t captured : 1;
  };
};

template <bool returnInfo>
__host__ __device__ inline MoveInfo makeMove(Board& board, Move move) {
  Board oldBoard = board;
  uint64_t one = 1;
  Region toRegionIdx = squareToRegion(static_cast<Square>(move.to));
  uint32_t toRegion = 1 << (REGION_SIZE - 1 - move.to % REGION_SIZE);
  uint32_t inHandOffsetForColor = 0;
  MoveInfo info;
  info.value = 0;
  bool captured = false;
  if (move.from < SQUARE_SIZE) {
    Region fromRegionIdx = squareToRegion(static_cast<Square>(move.from));
    uint32_t fromRegion = 1 << (REGION_SIZE - 1 - move.from % REGION_SIZE);
    // Check if white was captured
    if (board[BB::Type::ALL_WHITE][toRegionIdx] & toRegion) {
      inHandOffsetForColor = 7;
      captured = true;
#ifndef __CUDA_ARCH__
      info.toColor = 0;
#endif
    } else if (board[BB::Type::ALL_BLACK][toRegionIdx] & toRegion) {
      captured = true;
#ifndef __CUDA_ARCH__
      info.toColor = 1;
#endif
    }
    if (captured) {
      for (int i = 0; i < BB::Type::SIZE; i++) {
        if (board[static_cast<BB::Type>(i)][toRegionIdx] & toRegion) {
#ifndef __CUDA_ARCH__
          if (returnInfo) {
            if (i < BB::Type::PROMOTED) {
              info.toType = i;
              info.captured = 1;
            } else if (i == BB::Type::PROMOTED)
              info.toPromotion = 1;
          }
#endif
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
#ifndef __CUDA_ARCH__
        if (returnInfo) {
          if (i < BB::Type::PROMOTED)
            info.fromType = i;
          else
            info.fromColor = i - BB::Type::ALL_WHITE;
        }
#endif
        board[static_cast<BB::Type>(i)][fromRegionIdx] &= ~fromRegion;
        board[static_cast<BB::Type>(i)][toRegionIdx] |= toRegion;
      }
    }

    if (move.promotion) {
      board[BB::Type::PROMOTED][toRegionIdx] |= toRegion;
#ifndef __CUDA_ARCH__
      if (returnInfo)
        info.fromPromotion = 1;
#endif
    }
  } else {
    int offset = move.from - WHITE_PAWN_DROP;
    uint64_t addedValue = one << (4 * offset);
    board.inHand.value -= addedValue;
    board[static_cast<BB::Type>(offset % 7)][toRegionIdx] |= toRegion;
    board[static_cast<BB::Type>(offset / 7 + BB::Type::ALL_WHITE)][toRegionIdx] |= toRegion;
  }

  if (popcount(board[BB::Type::KING][TOP]) +
          popcount(board[BB::Type::KING][MID]) +
              popcount(board[BB::Type::KING][BOTTOM]) < 2) {
    printf("err\n");
  }
  return info;
}

__host__ __device__ inline MoveInfo makeMove(Board& board, Move move) {
  return makeMove<false>(board, move);
}

__host__ inline void unmakeMove(Board& board,
                                Move move,
                                MoveInfo moveReturnInfo) {
  uint64_t one = 1;
  Region toRegionIdx = squareToRegion(static_cast<Square>(move.to));
  uint32_t toRegion = 1 << (REGION_SIZE - 1 - move.to % REGION_SIZE);
  if (move.from >= SQUARE_SIZE) {
    int offset = move.from - WHITE_PAWN_DROP;
    uint64_t addedValue = one << (4 * offset);
    board.inHand.value += addedValue;
    board[static_cast<BB::Type>(offset % 7)][toRegionIdx] &= ~toRegion;
    board[static_cast<BB::Type>(offset / 7 + BB::Type::ALL_WHITE)][toRegionIdx] &= ~toRegion;
  } else {
    Region fromRegionIdx = squareToRegion(static_cast<Square>(move.from));
    uint32_t fromRegion = 1 << (REGION_SIZE - 1 - move.from % REGION_SIZE);
    board[static_cast<BB::Type>(moveReturnInfo.fromType)][toRegionIdx] &=
        ~toRegion;
    board[static_cast<BB::Type>(moveReturnInfo.fromType)][fromRegionIdx] |=
        fromRegion;
    board[static_cast<BB::Type>(BB::Type::ALL_WHITE + moveReturnInfo.fromColor)]
         [toRegionIdx] &= ~toRegion;
    board[static_cast<BB::Type>(BB::Type::ALL_WHITE + moveReturnInfo.fromColor)]
         [fromRegionIdx] |= fromRegion;
    if (moveReturnInfo.fromPromotion) {
      board[BB::Type::PROMOTED][toRegionIdx] &= ~toRegion;
    }

    if (moveReturnInfo.captured) {
      uint64_t addedValue = one << (4 * ((moveReturnInfo.fromColor ? 7 : 0) +
                                         moveReturnInfo.toType));
      board.inHand.value -= addedValue;
      board[static_cast<BB::Type>(moveReturnInfo.toType)][toRegionIdx] |=
          toRegion;
      board[static_cast<BB::Type>(BB::Type::ALL_WHITE + moveReturnInfo.toColor)]
           [toRegionIdx] |= toRegion;
      if (moveReturnInfo.toPromotion) {
        board[BB::Type::PROMOTED][toRegionIdx] |= toRegion;
      }
    }
  }
}
}  // namespace engine
}  // namespace shogi