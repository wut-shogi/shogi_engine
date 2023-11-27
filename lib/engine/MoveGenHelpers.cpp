#include "MoveGenHelpers.h"
#include <algorithm>

// Attack bitboards
std::array<std::array<Bitboard, BOARD_SIZE>, 128> initRankAttacks();
std::array<std::array<Bitboard, BOARD_SIZE>, 128> initFileAttacks();
std::array<std::array<Bitboard, BOARD_SIZE>, 128> initDiagRightAttacks();
std::array<std::array<Bitboard, BOARD_SIZE>, 128> initDiagLeftAttacks();
// Lance masks
std::array<Bitboard, BOARD_DIM> initWhiteLanceMasks();
std::array<Bitboard, BOARD_DIM> initBlackLanceMasks();

static std::array<std::array<Bitboard, BOARD_SIZE>, 128> rankAttacks =
    initRankAttacks();
static std::array<std::array<Bitboard, BOARD_SIZE>, 128> fileAttacks =
    initFileAttacks();
static std::array<std::array<Bitboard, BOARD_SIZE>, 128> diagRightAttacks =
    initDiagRightAttacks();
static std::array<std::array<Bitboard, BOARD_SIZE>, 128> diagLeftAttacks =
    initDiagLeftAttacks();

static std::array<Bitboard, BOARD_DIM> whiteLanceMasks = initWhiteLanceMasks();
static std::array<Bitboard, BOARD_DIM> blackLanceMasks = initBlackLanceMasks();

// Block patterns
static uint32_t getRankBlockPattern(const Bitboard& bb, Square square) {
  const uint32_t& region = bb[squareToRegion(square)];
  uint32_t rowsBeforeInRegion = (square / BOARD_DIM) % 3;
  uint32_t result = region << 5 << (rowsBeforeInRegion * BOARD_DIM) << 1 >> 25;
  return result;
}

static uint32_t getFileBlockPattern(const Bitboard& bbRot90, Square square) {
  return getRankBlockPattern(bbRot90, square);
}

static uint32_t getDiagRightBlockPattern(const Bitboard& bbRot45Right,
                                         Square square) {
  static const Region regionIdx[BOARD_SIZE] = {
      MID, TOP, TOP,    TOP,    TOP,    TOP,    TOP,    MID,    MID,     //
      TOP, TOP, TOP,    TOP,    TOP,    TOP,    MID,    MID,    MID,     //
      TOP, TOP, TOP,    TOP,    TOP,    MID,    MID,    MID,    BOTTOM,  //
      TOP, TOP, TOP,    TOP,    MID,    MID,    MID,    BOTTOM, BOTTOM,  //
      TOP, TOP, TOP,    MID,    MID,    MID,    BOTTOM, BOTTOM, BOTTOM,  //
      TOP, TOP, MID,    MID,    MID,    BOTTOM, BOTTOM, BOTTOM, BOTTOM,  //
      TOP, MID, MID,    MID,    BOTTOM, BOTTOM, BOTTOM, BOTTOM, BOTTOM,  //
      MID, MID, MID,    BOTTOM, BOTTOM, BOTTOM, BOTTOM, BOTTOM, BOTTOM,  //
      MID, MID, BOTTOM, BOTTOM, BOTTOM, BOTTOM, BOTTOM, BOTTOM, MID,
  };

  static const uint32_t shiftRight[BOARD_SIZE] = {
      18, 25, 22, 18, 13, 7,  0,  19, 9,   //
      25, 22, 18, 13, 7,  0,  19, 9,  1,   //
      22, 18, 13, 7,  0,  19, 9,  1,  20,  //
      18, 13, 7,  0,  19, 9,  1,  20, 14,  //
      13, 7,  0,  19, 9,  1,  20, 14, 9,   //
      7,  0,  19, 9,  1,  20, 14, 9,  5,   //
      0,  19, 9,  1,  20, 14, 9,  5,  2,   //
      19, 9,  1,  20, 14, 9,  5,  2,  0,   //
      9,  1,  20, 14, 9,  5,  2,  0,  0,
  };

  static const uint32_t mask[BOARD_SIZE] = {
      1,   3,   7,   15,  31,  63,  127, 255, 511,  //
      3,   7,   15,  31,  63,  127, 255, 511, 255,  //
      7,   15,  31,  63,  127, 255, 511, 255, 127,  //
      15,  31,  63,  127, 255, 511, 255, 127, 63,   //
      31,  63,  127, 255, 511, 255, 127, 63,  31,   //
      63,  127, 255, 511, 255, 127, 63,  31,  15,   //
      127, 255, 511, 255, 127, 63,  31,  15,  7,    //
      255, 511, 255, 127, 63,  31,  15,  7,   3,    //
      511, 255, 127, 63,  31,  15,  7,   3,   1,
  };

  const uint32_t& region = bbRot45Right[regionIdx[square]];
  uint32_t aftershift = region >> shiftRight[square] >> 1;
  uint32_t value = aftershift & (mask[square] / 4);
  return value;
}
static uint32_t getDiagLeftBlockPattern(const Bitboard& bbRot45Left,
                                        Square square) {
  static const Region regionIdx[BOARD_SIZE] = {
      MID,    MID,    TOP,    TOP,    TOP,    TOP,    TOP,    TOP, MID,  //
      MID,    MID,    MID,    TOP,    TOP,    TOP,    TOP,    TOP, TOP,  //
      BOTTOM, MID,    MID,    MID,    TOP,    TOP,    TOP,    TOP, TOP,  //
      BOTTOM, BOTTOM, MID,    MID,    MID,    TOP,    MID,    TOP, TOP,  //
      BOTTOM, BOTTOM, BOTTOM, MID,    MID,    MID,    TOP,    TOP, TOP,  //
      BOTTOM, BOTTOM, BOTTOM, BOTTOM, MID,    MID,    MID,    TOP, TOP,  //
      BOTTOM, BOTTOM, BOTTOM, BOTTOM, BOTTOM, MID,    MID,    MID, TOP,  //
      BOTTOM, BOTTOM, BOTTOM, BOTTOM, BOTTOM, BOTTOM, MID,    MID, MID,  //
      MID,    BOTTOM, BOTTOM, BOTTOM, BOTTOM, BOTTOM, BOTTOM, MID, MID,
  };

  static const uint32_t shiftRight[BOARD_SIZE] = {
      9,  19, 0,  7,  13, 18, 22, 25, 18,  //
      1,  9,  19, 0,  7,  13, 18, 22, 25,  //
      20, 1,  9,  19, 0,  7,  13, 18, 22,  //
      14, 20, 1,  9,  19, 0,  7,  13, 18,  //
      9,  14, 20, 1,  9,  19, 0,  7,  13,  //
      5,  9,  14, 20, 1,  9,  19, 0,  7,   //
      2,  5,  9,  14, 20, 1,  9,  19, 0,   //
      0,  2,  5,  9,  14, 20, 1,  9,  19,  //
      0,  0,  2,  5,  9,  14, 20, 1,  9,
  };

  static const uint32_t mask[BOARD_SIZE] = {
      511, 255, 127, 63,  31,  15,  7,   3,   1,    //
      255, 511, 255, 127, 63,  31,  15,  7,   3,    //
      127, 255, 511, 255, 127, 63,  31,  15,  7,    //
      63,  127, 255, 511, 255, 127, 63,  31,  15,   //
      31,  63,  127, 255, 511, 255, 127, 63,  31,   //
      15,  31,  63,  127, 255, 511, 255, 127, 63,   //
      7,   15,  31,  63,  127, 255, 511, 255, 127,  //
      3,   7,   15,  31,  63,  127, 255, 511, 255,  //
      1,   3,   7,   15,  31,  63,  127, 255, 511,
  };

  const uint32_t& region = bbRot45Left[regionIdx[square]];
  uint32_t aftershift = region >> shiftRight[square] >> 1;
  uint32_t value = aftershift & (mask[square] / 4);
  return value;
}


std::array<bool, 9> blockPatternToRow(uint32_t blockPattern) {
  uint32_t block = blockPattern << 1;
  std::array<bool, 9> result;
  result.fill(0);
  uint32_t mask = 1 << 7;
  for (uint32_t i = 1; i < 8; i++) {
    if (block & mask)
      result[i] = 1;
    mask = mask >> 1;
  }
  return result;
}

std::array<std::array<Bitboard, BOARD_SIZE>, 128> initRankAttacks() {
  std::array<std::array<Bitboard, BOARD_SIZE>, 128> result;
  std::array<bool, BOARD_SIZE> mat;

  for (uint32_t i = 0; i < BOARD_SIZE; i++) {
    uint32_t columnIdx = i % BOARD_DIM;
    uint32_t rowIdx = i / BOARD_DIM;
    for (uint32_t blockPattern = 0; blockPattern < 128; blockPattern++) {
      mat.fill(0);

      std::array<bool, BOARD_DIM> blockRow = blockPatternToRow(blockPattern);
      uint32_t lastLeft = 0;
      uint32_t firstRight = BOARD_DIM - 1;
      for (uint32_t col = 0; col < columnIdx; col++) {
        if (blockRow[col]) {
          lastLeft = col;
        }
      }
      for (uint32_t col = columnIdx + 1; col < blockRow.size(); col++) {
        if (blockRow[col]) {
          firstRight = col;
          break;
        }
      }
      // Fill first and last, becuase this moves are always valid
      for (uint32_t blockIdx = 0; blockIdx < blockRow.size(); blockIdx++) {
        if (blockIdx != columnIdx &&
            ((blockIdx >= lastLeft && blockIdx < columnIdx) ||
             (blockIdx <= firstRight && blockIdx > columnIdx)))
          mat[rowIdx * BOARD_DIM + blockIdx] = 1;
      }
      result[blockPattern][i] = mat;
    }
  }

  return result;
}

std::array<std::array<Bitboard, BOARD_SIZE>, 128> initFileAttacks() {
  std::array<std::array<Bitboard, BOARD_SIZE>, 128> result;

  for (uint32_t i = 0; i < BOARD_SIZE; i++) {
    for (uint32_t blockPattern = 0; blockPattern < 128; blockPattern++) {
      result[blockPattern][Rotate90AntiClockwise(static_cast<Square>(i))] =
          Rotate90Clockwise(rankAttacks[blockPattern][i]);
    }
  }

  return result;
}

std::array<std::array<Bitboard, BOARD_SIZE>, 128> initDiagRightAttacks() {
  std::array<std::array<Bitboard, BOARD_SIZE>, 128> result;
  std::array<bool, BOARD_SIZE> mat;

  for (uint32_t i = 0; i < BOARD_SIZE; i++) {
    uint32_t columnIdx = i % BOARD_DIM;
    uint32_t rowIdx = i / BOARD_DIM;
    int diagLength = (columnIdx + rowIdx < BOARD_DIM)
                         ? columnIdx + rowIdx + 1
                         : (BOARD_DIM - columnIdx) + (BOARD_DIM - rowIdx) - 1;
    uint32_t maxPattern = 1 << std::max((diagLength - 2), 0);
    for (uint32_t blockPattern = 0; blockPattern < maxPattern; blockPattern++) {
      mat.fill(0);
      std::array<bool, BOARD_DIM> blockRow =
          blockPatternToRow(blockPattern << (BOARD_DIM - diagLength));
      uint32_t tmpColIdx = columnIdx;
      uint32_t tmpRowIdx = rowIdx;
      uint32_t idxInDiag = 0;
      while (tmpRowIdx < BOARD_DIM - 1 && tmpColIdx > 0) {
        tmpRowIdx++;
        tmpColIdx--;
        idxInDiag++;
      }
      uint32_t lastLeft = 0;
      uint32_t firstRight = diagLength - 1;
      for (uint32_t col = lastLeft; col < idxInDiag; col++) {
        if (blockRow[col]) {
          lastLeft = col;
        }
      }
      for (uint32_t col = idxInDiag + 1; col < firstRight; col++) {
        if (blockRow[col]) {
          firstRight = col;
          break;
        }
      }

      for (uint32_t blockIdx = 0; blockIdx < blockRow.size(); blockIdx++) {
        if (blockIdx != idxInDiag &&
            ((blockIdx >= lastLeft && blockIdx < idxInDiag) ||
             (blockIdx <= firstRight && blockIdx > idxInDiag)))
          mat[tmpRowIdx * BOARD_DIM + tmpColIdx] = 1;
        tmpRowIdx--;
        tmpColIdx++;
        if (tmpRowIdx < 0 || tmpColIdx > BOARD_DIM - 1) {
          break;
        }
      }
      result[blockPattern][i] = mat;
    }
  }

  return result;
}

std::array<std::array<Bitboard, BOARD_SIZE>, 128> initDiagLeftAttacks() {
  std::array<std::array<Bitboard, BOARD_SIZE>, 128> result;
  std::array<bool, BOARD_SIZE> mat;

  for (uint32_t i = 0; i < BOARD_SIZE; i++) {
    uint32_t columnIdx = i % BOARD_DIM;
    uint32_t rowIdx = i / BOARD_DIM;
    int diagLength = ((BOARD_DIM - columnIdx) + rowIdx < BOARD_DIM)
                         ? (BOARD_DIM - columnIdx) + rowIdx
                         : columnIdx + (BOARD_DIM - rowIdx);
    uint32_t maxPattern = 1 << std::max((diagLength - 2), 0);
    for (uint32_t blockPattern = 0; blockPattern < maxPattern; blockPattern++) {
      mat.fill(0);
      std::array<bool, BOARD_DIM> blockRow =
          blockPatternToRow(blockPattern << (BOARD_DIM - diagLength));
      uint32_t tmpColIdx = columnIdx;
      uint32_t tmpRowIdx = rowIdx;
      uint32_t idxInDiag = 0;
      while (tmpRowIdx > 0 && tmpColIdx > 0) {
        tmpRowIdx--;
        tmpColIdx--;
        idxInDiag++;
      }
      uint32_t lastLeft = 0;
      uint32_t firstRight = diagLength - 1;
      for (uint32_t col = lastLeft; col < idxInDiag; col++) {
        if (blockRow[col]) {
          lastLeft = col;
        }
      }
      for (uint32_t col = idxInDiag + 1; col < firstRight; col++) {
        if (blockRow[col]) {
          firstRight = col;
          break;
        }
      }

      for (uint32_t blockIdx = 0; blockIdx < blockRow.size(); blockIdx++) {
        if (blockIdx != idxInDiag &&
            ((blockIdx >= lastLeft && blockIdx < idxInDiag) ||
             (blockIdx <= firstRight && blockIdx > idxInDiag)))
          mat[tmpRowIdx * BOARD_DIM + tmpColIdx] = 1;
        tmpRowIdx++;
        tmpColIdx++;
        if (tmpRowIdx < 0 || tmpColIdx > BOARD_DIM - 1) {
          break;
        }
      }
      result[blockPattern][i] = mat;
    }
  }

  return result;
}

std::array<Bitboard, BOARD_DIM> initWhiteLanceMasks() {
  std::array<Bitboard, BOARD_DIM> result;
  std::array<bool, BOARD_SIZE> mat;
  mat.fill(0);

  for (int i = BOARD_DIM - 1; i >= 0; i--) {
    result[i] = mat;
    mat[i * BOARD_DIM] = 1;
    mat[(i + 1) * BOARD_DIM - 1] = 1;
  }
  return result;
}
std::array<Bitboard, BOARD_DIM> initBlackLanceMasks() {
  std::array<Bitboard, BOARD_DIM> result;
  std::array<bool, BOARD_SIZE> mat;
  mat.fill(0);

  for (int i = 0; i < BOARD_DIM; i++) {
    result[i] = mat;
    mat[i * BOARD_DIM] = 1;
    mat[(i + 1) * BOARD_DIM - 1] = 1;
  }
  return result;
}

Bitboard moveN(Bitboard bb) {
  Bitboard out;
  out[TOP] =
      ((bb[TOP] << BOARD_DIM) | (bb[MID] >> (2 * BOARD_DIM))) & FULL_REGION;
  out[MID] =
      ((bb[MID] << BOARD_DIM) | (bb[BOTTOM] >> (2 * BOARD_DIM))) & FULL_REGION;
  out[BOTTOM] = (bb[BOTTOM] << BOARD_DIM) & FULL_REGION;
  return out;
}

Bitboard moveNE(Bitboard bb) {
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

Bitboard moveE(Bitboard bb) {
  Bitboard out;
  out[TOP] = (bb[TOP] & NOT_RIGHT_FILE) >> 1;
  out[MID] = (bb[MID] & NOT_RIGHT_FILE) >> 1;
  out[BOTTOM] = (bb[BOTTOM] & NOT_RIGHT_FILE) >> 1;
  return out;
}

Bitboard moveSE(Bitboard bb) {
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

Bitboard moveS(Bitboard bb) {
  Bitboard out;
  out[BOTTOM] =
      ((bb[BOTTOM] >> BOARD_DIM) | (bb[MID] << (2 * BOARD_DIM))) & FULL_REGION;
  out[MID] =
      ((bb[MID] >> BOARD_DIM) | (bb[TOP] << (2 * BOARD_DIM))) & FULL_REGION;
  out[TOP] = (bb[TOP] >> BOARD_DIM) & FULL_REGION;
  return out;
}

Bitboard moveSW(Bitboard bb) {
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

Bitboard moveW(Bitboard bb) {
  Bitboard out;
  out[TOP] = (bb[TOP] & NOT_LEFT_FILE) << 1;
  out[MID] = (bb[MID] & NOT_LEFT_FILE) << 1;
  out[BOTTOM] = (bb[BOTTOM] & NOT_LEFT_FILE) << 1;
  return out;
}

Bitboard moveNW(Bitboard bb) {
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

size_t countWhitePawnMoves(const Bitboard pawns, const Bitboard& validMoves) {
  Bitboard moved = moveS(pawns) & validMoves;
  return std::popcount<uint32_t>(moved[TOP]) +
         std::popcount<uint32_t>(moved[MID]) +
         std::popcount<uint32_t>(moved[BOTTOM] & (~BOTTOM_RANK)) * 2 +
         std::popcount<uint32_t>(moved[BOTTOM] & BOTTOM_RANK);
}

size_t countBlackPawnMoves(const Bitboard pawns, const Bitboard& validMoves) {
  Bitboard moved = moveN(pawns) & validMoves;
  return std::popcount<uint32_t>(moved[BOTTOM]) +
         std::popcount<uint32_t>(moved[MID]) +
         std::popcount<uint32_t>(moved[TOP] & (~TOP_RANK)) * 2 +
         std::popcount<uint32_t>(moved[TOP] & TOP_RANK);
}

size_t countWhiteKnightMoves(const Bitboard knights,
                             const Bitboard& validMoves) {
  // SSE move
  Bitboard moved = moveS(moveSE(knights)) & validMoves;
  int moveCount = std::popcount<uint32_t>(moved[TOP]) +
                  std::popcount<uint32_t>(moved[MID]) +
                  std::popcount<uint32_t>(moved[BOTTOM] & (TOP_RANK)) * 2 +
                  std::popcount<uint32_t>(moved[BOTTOM] & (~TOP_RANK));
  // SSW move
  moved = moveS(moveSW(knights)) & validMoves;
  return moveCount + std::popcount<uint32_t>(moved[TOP]) +
         std::popcount<uint32_t>(moved[MID]) +
         std::popcount<uint32_t>(moved[BOTTOM] & (TOP_RANK)) * 2 +
         std::popcount<uint32_t>(moved[BOTTOM] & (~TOP_RANK));
}

size_t countBlackKnightMoves(const Bitboard knights,
                             const Bitboard& validMoves) {
  // NNE move
  Bitboard moved = moveN(moveNE(knights)) & validMoves;
  int moveCount = std::popcount<uint32_t>(moved[BOTTOM]) +
                  std::popcount<uint32_t>(moved[MID]) +
                  std::popcount<uint32_t>(moved[TOP] & (BOTTOM_RANK)) * 2 +
                  std::popcount<uint32_t>(moved[TOP] & (~BOTTOM_RANK));
  // NNW move
  moved = moveN(moveNW(knights)) & validMoves;
  return moveCount + std::popcount<uint32_t>(moved[BOTTOM]) +
         std::popcount<uint32_t>(moved[MID]) +
         std::popcount<uint32_t>(moved[TOP] & (BOTTOM_RANK)) * 2 +
         std::popcount<uint32_t>(moved[TOP] & (~BOTTOM_RANK));
}

size_t countWhiteSilverGeneralMoves(const Bitboard silverGenerals,
                                    const Bitboard& validMoves) {
  // S Move
  Bitboard moved = moveS(silverGenerals) & validMoves;
  int moveCount = std::popcount<uint32_t>(moved[TOP]) +
                  std::popcount<uint32_t>(moved[MID]) +
                  std::popcount<uint32_t>(moved[BOTTOM]) * 2;
  // SE Move
  moved = moveSE(silverGenerals) & validMoves;
  moveCount += std::popcount<uint32_t>(moved[TOP]) +
               std::popcount<uint32_t>(moved[MID]) +
               std::popcount<uint32_t>(moved[BOTTOM]) * 2;

  // SW Move
  moved = moveSW(silverGenerals) & validMoves;
  moveCount += std::popcount<uint32_t>(moved[TOP]) +
               std::popcount<uint32_t>(moved[MID]) +
               std::popcount<uint32_t>(moved[BOTTOM]) * 2;

  // NE Move
  moved = moveNE(silverGenerals) & validMoves;
  moveCount += std::popcount<uint32_t>(moved[TOP]) +
               std::popcount<uint32_t>(moved[MID]) +
               std::popcount<uint32_t>(moved[BOTTOM]) * 2;

  // NW Move
  moved = moveNW(silverGenerals) & validMoves;
  moveCount += std::popcount<uint32_t>(moved[TOP]) +
               std::popcount<uint32_t>(moved[MID]) +
               std::popcount<uint32_t>(moved[BOTTOM]) * 2;

  return moveCount;
}

size_t countBlackSilverGeneralMoves(const Bitboard silverGenerals,
                                    const Bitboard& validMoves) {
  // N Move
  Bitboard moved = moveN(silverGenerals) & validMoves;
  int moveCount = std::popcount<uint32_t>(moved[BOTTOM]) +
                  std::popcount<uint32_t>(moved[MID]) +
                  std::popcount<uint32_t>(moved[TOP]) * 2;
  // NE Move
  moved = moveNE(silverGenerals) & validMoves;
  moveCount += std::popcount<uint32_t>(moved[BOTTOM]) +
               std::popcount<uint32_t>(moved[MID]) +
               std::popcount<uint32_t>(moved[TOP]) * 2;

  // SE Move
  moved = moveSE(silverGenerals) & validMoves;
  moveCount += std::popcount<uint32_t>(moved[BOTTOM]) +
               std::popcount<uint32_t>(moved[MID]) +
               std::popcount<uint32_t>(moved[TOP]) * 2;

  // SW Move
  moved = moveSW(silverGenerals) & validMoves;
  moveCount += std::popcount<uint32_t>(moved[BOTTOM]) +
               std::popcount<uint32_t>(moved[MID]) +
               std::popcount<uint32_t>(moved[TOP]) * 2;

  // NW Move
  moved = moveNW(silverGenerals) & validMoves;
  moveCount += std::popcount<uint32_t>(moved[BOTTOM]) +
               std::popcount<uint32_t>(moved[MID]) +
               std::popcount<uint32_t>(moved[TOP]) * 2;

  return moveCount;
}

size_t countWhiteGoldGeneralMoves(const Bitboard goldGenerals,
                                  const Bitboard& validMoves) {
  // S Move
  Bitboard moved = moveS(goldGenerals) & validMoves;
  int moveCount = std::popcount<uint32_t>(moved[BOTTOM]) +
                  std::popcount<uint32_t>(moved[MID]) +
                  std::popcount<uint32_t>(moved[TOP]);
  // SE Move
  moved = moveSE(goldGenerals) & validMoves;
  moveCount += std::popcount<uint32_t>(moved[BOTTOM]) +
               std::popcount<uint32_t>(moved[MID]) +
               std::popcount<uint32_t>(moved[TOP]);

  // SW Move
  moved = moveSW(goldGenerals) & validMoves;
  moveCount += std::popcount<uint32_t>(moved[BOTTOM]) +
               std::popcount<uint32_t>(moved[MID]) +
               std::popcount<uint32_t>(moved[TOP]);

  // E Move
  moved = moveE(goldGenerals) & validMoves;
  moveCount += std::popcount<uint32_t>(moved[BOTTOM]) +
               std::popcount<uint32_t>(moved[MID]) +
               std::popcount<uint32_t>(moved[TOP]);

  // W Move
  moved = moveW(goldGenerals) & validMoves;
  moveCount += std::popcount<uint32_t>(moved[BOTTOM]) +
               std::popcount<uint32_t>(moved[MID]) +
               std::popcount<uint32_t>(moved[TOP]);

  // N Move
  moved = moveN(goldGenerals) & validMoves;
  moveCount += std::popcount<uint32_t>(moved[BOTTOM]) +
               std::popcount<uint32_t>(moved[MID]) +
               std::popcount<uint32_t>(moved[TOP]);

  return moveCount;
}

size_t countBlackGoldGeneralMoves(const Bitboard goldGenerals,
                                  const Bitboard& validMoves) {
  // N Move
  Bitboard moved = moveN(goldGenerals) & validMoves;
  int moveCount = std::popcount<uint32_t>(moved[BOTTOM]) +
                  std::popcount<uint32_t>(moved[MID]) +
                  std::popcount<uint32_t>(moved[TOP]);
  // NE Move
  moved = moveNE(goldGenerals) & validMoves;
  moveCount += std::popcount<uint32_t>(moved[BOTTOM]) +
               std::popcount<uint32_t>(moved[MID]) +
               std::popcount<uint32_t>(moved[TOP]);

  // NW Move
  moved = moveNW(goldGenerals) & validMoves;
  moveCount += std::popcount<uint32_t>(moved[BOTTOM]) +
               std::popcount<uint32_t>(moved[MID]) +
               std::popcount<uint32_t>(moved[TOP]);

  // E Move
  moved = moveE(goldGenerals) & validMoves;
  moveCount += std::popcount<uint32_t>(moved[BOTTOM]) +
               std::popcount<uint32_t>(moved[MID]) +
               std::popcount<uint32_t>(moved[TOP]);

  // W Move
  moved = moveW(goldGenerals) & validMoves;
  moveCount += std::popcount<uint32_t>(moved[BOTTOM]) +
               std::popcount<uint32_t>(moved[MID]) +
               std::popcount<uint32_t>(moved[TOP]);

  // S Move
  moved = moveS(goldGenerals) & validMoves;
  moveCount += std::popcount<uint32_t>(moved[BOTTOM]) +
               std::popcount<uint32_t>(moved[MID]) +
               std::popcount<uint32_t>(moved[TOP]);

  return moveCount;
}

size_t countKingMoves(const Bitboard king, const Bitboard& validMoves) {
  Bitboard moved = (moveN(king) | moveNE(king) | moveE(king) | moveSE(king) |
                    moveS(king) | moveSW(king) | moveW(king) | moveNW(king)) &
                   validMoves;
  return std::popcount<uint32_t>(moved[BOTTOM]) +
         std::popcount<uint32_t>(moved[MID]) +
         std::popcount<uint32_t>(moved[TOP]);
}

size_t countWhiteLanceMoves(const Bitboard lances,
                            const Bitboard& validMoves,
                            const Bitboard& occupiedRot90) {
  int moveCount = 0;
  int numberOfLances = 0;
  BitboardIterator iterator(lances);
  while (numberOfLances < 2 && iterator.Next()) {
    if (iterator.IsCurrentSquareOccupied()) {
      numberOfLances++;
      Bitboard moved =
          fileAttacks[getFileBlockPattern(occupiedRot90,
                                          iterator.GetCurrentSquare())]
                     [iterator.GetCurrentSquare()] &
          whiteLanceMasks[squareToRank(iterator.GetCurrentSquare())] &
          validMoves;

      moveCount += std::popcount<uint32_t>(moved[TOP]) +
                   std::popcount<uint32_t>(moved[MID]) +
                   std::popcount<uint32_t>(moved[BOTTOM] & (~BOTTOM_RANK)) * 2 +
                   std::popcount<uint32_t>(moved[BOTTOM] & BOTTOM_RANK);
    }
  }
  return moveCount;
}

size_t countBlackLanceMoves(const Bitboard lances,
                            const Bitboard& validMoves,
                            const Bitboard& occupiedRot90) {
  int moveCount = 0;
  int numberOfLances = 0;
  BitboardIterator iterator(lances);
  while (numberOfLances < 2 && iterator.Next()) {
    if (iterator.IsCurrentSquareOccupied()) {
      numberOfLances++;
      Bitboard moved =
          fileAttacks[getFileBlockPattern(occupiedRot90,
                                          iterator.GetCurrentSquare())]
                     [iterator.GetCurrentSquare()] &
          blackLanceMasks[squareToRank(iterator.GetCurrentSquare())] &
          validMoves;

      moveCount += std::popcount<uint32_t>(moved[BOTTOM]) +
                   std::popcount<uint32_t>(moved[MID]) +
                   std::popcount<uint32_t>(moved[TOP] & (~BOTTOM_RANK)) * 2 +
                   std::popcount<uint32_t>(moved[TOP] & BOTTOM_RANK);
    }
  }
  return moveCount;
}

size_t countWhiteBishopMoves(const Bitboard bishops,
                             const Bitboard& validMoves,
                             const Bitboard& occupiedRot45Right,
                             const Bitboard& occupiedRot45Left) {
  int moveCount = 0;
  int numberOfBishops = 0;
  BitboardIterator iterator(bishops);
  while (numberOfBishops < 1 && iterator.Next()) {
    if (iterator.IsCurrentSquareOccupied()) {
      numberOfBishops++;
      Square square = iterator.GetCurrentSquare();
      Bitboard moved = (diagRightAttacks[getDiagRightBlockPattern(
                            occupiedRot45Right, square)][square] |
                        diagLeftAttacks[getDiagLeftBlockPattern(
                            occupiedRot45Left, square)][square]) &
                       validMoves;

      int multiplier = squareToRank(square) > 5 ? 2 : 1;
      moveCount += (std::popcount<uint32_t>(moved[TOP]) +
                    std::popcount<uint32_t>(moved[MID])) *
                       multiplier +
                   std::popcount<uint32_t>(moved[BOTTOM]) * 2;
    }
  }
  return moveCount;
}

size_t countBlackBishopMoves(const Bitboard bishops,
                             const Bitboard& validMoves,
                             const Bitboard& occupiedRot45Right,
                             const Bitboard& occupiedRot45Left) {
  int moveCount = 0;
  int numberOfBishops = 0;
  BitboardIterator iterator(bishops);
  while (numberOfBishops < 1 && iterator.Next()) {
    if (iterator.IsCurrentSquareOccupied()) {
      numberOfBishops++;
      Square square = iterator.GetCurrentSquare();
      Bitboard moved = (diagRightAttacks[getDiagRightBlockPattern(
                            occupiedRot45Right, square)][square] |
                        diagLeftAttacks[getDiagLeftBlockPattern(
                            occupiedRot45Left, square)][square]) &
                       validMoves;

      int multiplier = squareToRank(square) < 3 ? 2 : 1;
      moveCount += (std::popcount<uint32_t>(moved[BOTTOM]) +
                    std::popcount<uint32_t>(moved[MID])) *
                       multiplier +
                   std::popcount<uint32_t>(moved[TOP]) * 2;
    }
  }
  return moveCount;
}

size_t countWhiteRookMoves(const Bitboard rooks,
                           const Bitboard& validMoves,
                           const Bitboard& occupied,
                           const Bitboard& occupiedRot90) {
  int moveCount = 0;
  int numberOfBishops = 0;
  BitboardIterator iterator(rooks);
  while (numberOfBishops < 1 && iterator.Next()) {
    if (iterator.IsCurrentSquareOccupied()) {
      numberOfBishops++;
      Square square = iterator.GetCurrentSquare();
      Bitboard moved =
          (fileAttacks[getFileBlockPattern(occupiedRot90, square)][square] |
           rankAttacks[getRankBlockPattern(occupied, square)][square]) &
          validMoves;

      int multiplier = squareToRank(square) > 5 ? 2 : 1;
      moveCount += (std::popcount<uint32_t>(moved[TOP]) +
                    std::popcount<uint32_t>(moved[MID])) *
                       multiplier +
                   std::popcount<uint32_t>(moved[BOTTOM]) * 2;
    }
  }
  return moveCount;
}

size_t countBlackRookMoves(const Bitboard rooks,
                           const Bitboard& validMoves,
                           const Bitboard& occupied,
                           const Bitboard& occupiedRot90) {
  int moveCount = 0;
  int numberOfBishops = 0;
  BitboardIterator iterator(rooks);
  while (numberOfBishops < 1 && iterator.Next()) {
    if (iterator.IsCurrentSquareOccupied()) {
      numberOfBishops++;
      Square square = iterator.GetCurrentSquare();
      Bitboard moved =
          (fileAttacks[getFileBlockPattern(occupiedRot90, square)][square] |
           rankAttacks[getRankBlockPattern(occupied, square)][square]) &
          validMoves;

      int multiplier = squareToRank(square) < 3 ? 2 : 1;
      moveCount += (std::popcount<uint32_t>(moved[BOTTOM]) +
                    std::popcount<uint32_t>(moved[MID])) *
                       multiplier +
                   std::popcount<uint32_t>(moved[TOP]) * 2;
    }
  }
  return moveCount;
}

size_t countHorseMoves(const Bitboard horse,
                       const Bitboard& validMoves,
                       const Bitboard& occupiedRot45Right,
                       const Bitboard& occupiedRot45Left) {
  int moveCount = 0;
  int numberOfBishops = 0;
  BitboardIterator iterator(horse);
  while (numberOfBishops < 1 && iterator.Next()) {
    if (iterator.IsCurrentSquareOccupied()) {
      numberOfBishops++;
      Square square = iterator.GetCurrentSquare();
      // Bishop-like moves
      Bitboard moved =
          (diagRightAttacks[getDiagRightBlockPattern(occupiedRot45Right,
                                                     square)][square] |
           diagLeftAttacks[getDiagLeftBlockPattern(occupiedRot45Left, square)]
                          [square] |
           moveN(horse) | moveE(horse) | moveS(horse) | moveW(horse)) &
          validMoves;

      moveCount += std::popcount<uint32_t>(moved[TOP]) +
                   std::popcount<uint32_t>(moved[MID]) +
                   std::popcount<uint32_t>(moved[BOTTOM]);
    }
  }
  return moveCount;
}


size_t countDragonMoves(const Bitboard dragon,
                        const Bitboard& validMoves,
                        const Bitboard& occupied,
                        const Bitboard& occupiedRot90) {
  int moveCount = 0;
  int numberOfBishops = 0;
  BitboardIterator iterator(dragon);
  while (numberOfBishops < 1 && iterator.Next()) {
    if (iterator.IsCurrentSquareOccupied()) {
      numberOfBishops++;
      Square square = iterator.GetCurrentSquare();
      // Bishop-like moves
      Bitboard moved =
          (fileAttacks[getFileBlockPattern(occupiedRot90, square)][square] |
           rankAttacks[getRankBlockPattern(occupied, square)][square] |
           moveNE(dragon) | moveNW(dragon) | moveSE(dragon) | moveSW(dragon)) &
          validMoves;

      moveCount += std::popcount<uint32_t>(moved[TOP]) +
                   std::popcount<uint32_t>(moved[MID]) +
                   std::popcount<uint32_t>(moved[BOTTOM]);
    }
  }
  return moveCount;
}

size_t countDropMoves(const NumberOfPieces& inHand,
    const Bitboard& freeSquares,
    const Bitboard ownPawns,
    const Bitboard enemyKing, bool isWhite) {
  int moveCount = 0;
  Bitboard legalDropSpots = freeSquares;
    // Pawns
  if (inHand.Pawn > 0) {
    // Cannot drop on last rank
    legalDropSpots[BOTTOM] &= ~BOTTOM_RANK;
    // Cannot drop to give checkmate
    legalDropSpots &= ~(isWhite ? moveN(enemyKing) : moveS(enemyKing));
    // Cannot drop on file with other pawn
    Bitboard validFiles;
    for (int fileIdx = 0; fileIdx < BOARD_DIM; fileIdx++) {
      Bitboard file = fileAttacks[0][(Square)fileIdx];
      if (empty(file & ownPawns)) {
        validFiles |= file;
      }
    }
    legalDropSpots &= validFiles;
    moveCount += std::popcount<uint32_t>(legalDropSpots[TOP]) +
                 std::popcount<uint32_t>(legalDropSpots[MID]) +
                 std::popcount<uint32_t>(legalDropSpots[BOTTOM]);
  }

  int otherPiecesCount = (inHand.Knight > 0) + (inHand.Lance > 0) +
                         (inHand.SilverGeneral > 0) + (inHand.GoldGeneral > 0) +
                         (inHand.Bishop > 0) + (inHand.Rook > 0);
  moveCount +=
      otherPiecesCount * (std::popcount<uint32_t>(legalDropSpots[TOP]) +
                          std::popcount<uint32_t>(legalDropSpots[MID]) +
                          std::popcount<uint32_t>(legalDropSpots[BOTTOM]));
  return moveCount;
}

size_t countBlackDropMoves(const InHand& inHand,
                           const Bitboard& freeSquares,
                           const Bitboard blackPawns,
                           const Bitboard whiteKing);