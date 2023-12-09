#include "MoveGenHelpers.h"
#include <algorithm>

namespace shogi {
namespace engine {
/// Static arrays initialization
// Attack bitboards
std::array<std::array<Bitboard, BOARD_SIZE>, 128> initRankAttacks();
std::array<std::array<Bitboard, BOARD_SIZE>, 128> initFileAttacks();
std::array<std::array<Bitboard, BOARD_SIZE>, 128> initDiagRightAttacks();
std::array<std::array<Bitboard, BOARD_SIZE>, 128> initDiagLeftAttacks();
// Lance masks
std::array<Bitboard, BOARD_DIM> initRankMask();
std::array<Bitboard, BOARD_DIM> initFileMask();

static std::array<std::array<Bitboard, BOARD_SIZE>, 128> rankAttacks =
    initRankAttacks();
static std::array<std::array<Bitboard, BOARD_SIZE>, 128> fileAttacks =
    initFileAttacks();
static std::array<std::array<Bitboard, BOARD_SIZE>, 128> diagRightAttacks =
    initDiagRightAttacks();
static std::array<std::array<Bitboard, BOARD_SIZE>, 128> diagLeftAttacks =
    initDiagLeftAttacks();

static std::array<Bitboard, BOARD_DIM> rankMask = initRankMask();
static std::array<Bitboard, BOARD_DIM> fileMask = initFileMask();

// Block patterns
static uint32_t getRankBlockPattern(const Bitboard& bb, Square square) {
  const uint32_t& region = bb[squareToRegion(square)];
  uint32_t rowsBeforeInRegion = (square / BOARD_DIM) % 3;
  uint32_t result = region << 5 << (rowsBeforeInRegion * BOARD_DIM) << 1 >> 25;
  return result;
}

static uint32_t getFileBlockPattern(const Bitboard& occupied, Square square) {
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

uint32_t getDiagRightBlockPattern(const Bitboard& occupied, Square square) {
  static const uint32_t startingSquare[BOARD_SIZE] = {
      0, 1,  2,  3,  4,  5,  6,  7,  8,   //
      1, 2,  3,  4,  5,  6,  7,  8,  17,  //
      2, 3,  4,  5,  6,  7,  8,  17, 26,  //
      3, 4,  5,  6,  7,  8,  17, 26, 35,  //
      4, 5,  6,  7,  8,  17, 26, 35, 44,  //
      5, 6,  7,  8,  17, 26, 35, 44, 53,  //
      6, 7,  8,  17, 26, 35, 44, 53, 62,  //
      7, 8,  17, 26, 35, 44, 53, 62, 71,  //
      8, 17, 26, 35, 44, 53, 62, 71, 80,
  };
  uint32_t result = 0;
  int len = startingSquare[square] > 9 ? 7 - startingSquare[square] / 9
                                       : startingSquare[square] - 1;
  for (int i = 0; i < len; i++) {
    result += occupied.GetBit(
                  static_cast<Square>(startingSquare[square] + i * SW + SW))
              << i;
  }
  return result;
}

uint32_t getDiagLeftBlockPattern(const Bitboard& occupied, Square square) {
  static const uint32_t startingSquare[BOARD_SIZE] = {
      0,  1,  2,  3,  4,  5,  6,  7, 8,  //
      9,  0,  1,  2,  3,  4,  5,  6, 7,  //
      18, 9,  0,  1,  2,  3,  4,  5, 6,  //
      27, 18, 9,  0,  1,  2,  3,  4, 5,  //
      36, 27, 18, 9,  0,  1,  2,  3, 4,  //
      45, 36, 27, 18, 9,  0,  1,  2, 3,  //
      54, 45, 36, 27, 18, 9,  0,  1, 2,  //
      63, 4,  45, 36, 27, 18, 9,  0, 1,  //
      72, 63, 4,  45, 36, 27, 18, 9, 0,
  };
  uint32_t result = 0;
  int len = startingSquare[square] > 9 ? 7 - startingSquare[square] / 9
                                       : 7 - startingSquare[square] % 9;
  for (int i = 0; i < len; i++) {
    result += occupied.GetBit(
                  static_cast<Square>(startingSquare[square] + i * SE + SE))
              << i;
  }
  return result;
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
      int lastLeft = 0;
      int firstRight = BOARD_DIM - 1;
      for (int col = 0; col < columnIdx; col++) {
        if (blockRow[col]) {
          lastLeft = col;
        }
      }
      for (int col = columnIdx + 1; col < blockRow.size(); col++) {
        if (blockRow[col]) {
          firstRight = col;
          break;
        }
      }
      for (int blockIdx = 0; blockIdx < blockRow.size(); blockIdx++) {
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
  std::array<bool, BOARD_SIZE> mat;

  for (uint32_t i = 0; i < BOARD_SIZE; i++) {
    uint32_t columnIdx = i % BOARD_DIM;
    uint32_t rowIdx = i / BOARD_DIM;
    for (uint32_t blockPattern = 0; blockPattern < 128; blockPattern++) {
      mat.fill(0);

      std::array<bool, BOARD_DIM> blockRow = blockPatternToRow(blockPattern);
      int lastLeft = 0;
      int firstRight = BOARD_DIM - 1;
      for (int row = 0; row < rowIdx; row++) {
        if (blockRow[row]) {
          lastLeft = row;
        }
      }
      for (int row = rowIdx + 1; row < blockRow.size(); row++) {
        if (blockRow[row]) {
          firstRight = row;
          break;
        }
      }
      for (int blockIdx = 0; blockIdx < blockRow.size(); blockIdx++) {
        if (blockIdx != rowIdx &&
            ((blockIdx >= lastLeft && blockIdx < rowIdx) ||
             (blockIdx <= firstRight && blockIdx > rowIdx)))
          mat[blockIdx * BOARD_DIM + columnIdx] = 1;
      }
      result[blockPattern][i] = mat;
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
      int tmpColIdx = columnIdx;
      int tmpRowIdx = rowIdx;
      int idxInDiag = 0;
      //// TODO naprawiæ chyba row z col zamienione albo coœ
      while (tmpColIdx > 0 && tmpRowIdx < BOARD_DIM - 1) {
        tmpRowIdx++;
        tmpColIdx--;
        idxInDiag++;
      }
      int lastLeft = 0;
      int firstRight = diagLength - 1;
      for (int col = lastLeft; col < idxInDiag; col++) {
        if (blockRow[col]) {
          lastLeft = col;
        }
      }
      for (int col = idxInDiag + 1; col < diagLength; col++) {
        if (blockRow[col]) {
          firstRight = col;
          break;
        }
      }

      for (int blockIdx = 0; blockIdx < blockRow.size(); blockIdx++) {
        if (blockIdx != idxInDiag &&
            ((blockIdx >= lastLeft && blockIdx < idxInDiag) ||
             (blockIdx <= firstRight && blockIdx > idxInDiag)))
          mat[tmpRowIdx * BOARD_DIM + tmpColIdx] = 1;
        if (tmpRowIdx == 0 || tmpColIdx == BOARD_DIM - 1) {
          break;
        }
        tmpRowIdx--;
        tmpColIdx++;
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
      int tmpColIdx = columnIdx;
      int tmpRowIdx = rowIdx;
      int idxInDiag = 0;
      while (tmpRowIdx < BOARD_DIM - 1 && tmpColIdx < BOARD_DIM - 1) {
        tmpRowIdx++;
        tmpColIdx++;
        idxInDiag++;
      }
      int lastLeft = 0;
      int firstRight = diagLength - 1;
      for (int col = lastLeft; col < idxInDiag; col++) {
        if (blockRow[col]) {
          lastLeft = col;
        }
      }
      for (int col = idxInDiag + 1; col < firstRight; col++) {
        if (blockRow[col]) {
          firstRight = col;
          break;
        }
      }

      for (int blockIdx = 0; blockIdx < blockRow.size(); blockIdx++) {
        if (blockIdx != idxInDiag &&
            ((blockIdx >= lastLeft && blockIdx < idxInDiag) ||
             (blockIdx <= firstRight && blockIdx > idxInDiag)))
          mat[tmpRowIdx * BOARD_DIM + tmpColIdx] = 1;
        if (tmpRowIdx == 0 || tmpColIdx == 0) {
          break;
        }
        tmpRowIdx--;
        tmpColIdx--;
      }
      result[blockPattern][i] = mat;
    }
  }

  return result;
}

std::array<Bitboard, BOARD_DIM> initRankMask() {
  std::array<Bitboard, BOARD_DIM> result;
  std::array<bool, BOARD_SIZE> mat;
  mat.fill(0);

  for (int i = 0; i < BOARD_DIM; i++) {
    for (int j = 0; j < BOARD_DIM; j++) {
      mat[i * BOARD_DIM + j] = 1;
    }
    result[i] = mat;
  }
  return result;
}

std::array<Bitboard, BOARD_DIM> initFileMask() {
  std::array<Bitboard, BOARD_DIM> result;
  std::array<bool, BOARD_SIZE> mat;
  mat.fill(0);

  for (int i = 0; i < BOARD_DIM; i++) {
    for (int j = 0; j < BOARD_DIM; j++) {
      mat[j * BOARD_DIM + i] = 1;
    }
    result[i] = mat;
  }
  return result;
}

const Bitboard& getRankAttacks(const Square& square, const Bitboard& occupied) {
  return rankAttacks[getRankBlockPattern(occupied, square)][square];
}

const Bitboard& getFileAttacks(const Square& square, const Bitboard& occupied) {
  return fileAttacks[getFileBlockPattern(occupied, square)][square];
}
const Bitboard& getDiagRightAttacks(const Square& square,
                                    const Bitboard& occupied) {
  return diagRightAttacks[getDiagRightBlockPattern(occupied, square)][square];
}
const Bitboard& getDiagLeftAttacks(const Square& square,
                                   const Bitboard& occupied) {
  return diagLeftAttacks[getDiagLeftBlockPattern(occupied, square)][square];
}
const Bitboard& getRankMask(const uint32_t& rank) {
  return rankMask[rank];
}
const Bitboard& getFileMask(const uint32_t& file) {
  return fileMask[file];
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

Bitboard getFullFile(int fileIdx) {
  uint32_t region = FIRST_FILE >> fileIdx;
  return Bitboard(region, region, region);
}
}  // namespace engine
}  // namespace shogi