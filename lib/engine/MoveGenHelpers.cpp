#include "MoveGenHelpers.h"
#include <algorithm>

/// Static arrays initialization
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

// Attack bitboards
void whitePawnsAttackBitboards(const Bitboard pawns,
                               Bitboard* outAttacksBitboards) {
  outAttacksBitboards[0] = moveS(pawns);
}
void blackPawnsAttackBitboards(const Bitboard pawns,
                               Bitboard* outAttacksBitboards) {
  outAttacksBitboards[0] = moveN(pawns);
}
void whiteKnightsAttackBitboards(const Bitboard knights,
                                 Bitboard* outAttacksBitboards) {
  outAttacksBitboards[0] = moveS(moveSE(knights));
  outAttacksBitboards[1] = moveS(moveSW(knights));
}
void blackKnightsAttackBitboards(const Bitboard knights,
                                 Bitboard* outAttacksBitboards) {
  outAttacksBitboards[0] = moveN(moveNE(knights));
  outAttacksBitboards[1] = moveN(moveNW(knights));
}
void whiteSilverGeneralsAttackBitboards(const Bitboard silverGenerals,
                                        Bitboard* outAttacksBitboards) {
  outAttacksBitboards[0] = moveNE(silverGenerals);
  outAttacksBitboards[1] = moveNW(silverGenerals);
  outAttacksBitboards[2] = moveSE(silverGenerals);
  outAttacksBitboards[3] = moveS(silverGenerals);
  outAttacksBitboards[4] = moveSW(silverGenerals);
}
void blackSilverGeneralsAttackBitboards(const Bitboard silverGenerals,
                                        Bitboard* outAttacksBitboards) {
  outAttacksBitboards[0] = moveSE(silverGenerals);
  outAttacksBitboards[1] = moveSW(silverGenerals);
  outAttacksBitboards[2] = moveNE(silverGenerals);
  outAttacksBitboards[3] = moveN(silverGenerals);
  outAttacksBitboards[4] = moveNW(silverGenerals);
}
void whiteGoldGeneralsAttackBitboards(const Bitboard goldGenerals,
                                      Bitboard* outAttacksBitboards) {
  outAttacksBitboards[0] = moveSE(goldGenerals);
  outAttacksBitboards[1] = moveS(goldGenerals);
  outAttacksBitboards[2] = moveSW(goldGenerals);
  outAttacksBitboards[3] = moveE(goldGenerals);
  outAttacksBitboards[4] = moveN(goldGenerals);
  outAttacksBitboards[5] = moveW(goldGenerals);
}
void blackGoldGeneralsAttackBitboards(const Bitboard goldGenerals,
                                      Bitboard* outAttacksBitboards) {
  outAttacksBitboards[0] = moveNE(goldGenerals);
  outAttacksBitboards[1] = moveN(goldGenerals);
  outAttacksBitboards[2] = moveNW(goldGenerals);
  outAttacksBitboards[3] = moveE(goldGenerals);
  outAttacksBitboards[4] = moveS(goldGenerals);
  outAttacksBitboards[5] = moveW(goldGenerals);
}
void kingAttackBitboards(const Square king, Bitboard* outAttacksBitboards) {
  Bitboard kingBB = Bitboard(king);
  outAttacksBitboards[0] = moveNE(kingBB);
  outAttacksBitboards[1] = moveN(kingBB);
  outAttacksBitboards[2] = moveNW(kingBB);
  outAttacksBitboards[3] = moveE(kingBB);
  outAttacksBitboards[4] = moveW(kingBB);
  outAttacksBitboards[5] = moveSE(kingBB);
  outAttacksBitboards[6] = moveS(kingBB);
  outAttacksBitboards[7] = moveSW(kingBB);
}
void whiteLanceAttackBitboards(const Square lance,
                               const Bitboard& occupiedRot90,
                               Bitboard* outAttacksBitboards) {
  if (lance == Square::NONE) {
    outAttacksBitboards[0] = Bitboard();
    return;
  }
  outAttacksBitboards[0] = fileAttacks[getFileBlockPattern(occupiedRot90, lance)][lance] &
              whiteLanceMasks[squareToRank(lance)];
}
void blackLanceAttackBitboards(const Square lance,
                               const Bitboard& occupiedRot90,
                               Bitboard* outAttacksBitboards) {
  if (lance == Square::NONE) {
    outAttacksBitboards[0] = Bitboard();
    return;
  }
  outAttacksBitboards[0] = fileAttacks[getFileBlockPattern(occupiedRot90, lance)][lance] &
                 blackLanceMasks[squareToRank(lance)];
}
void bishopAttackBitboards(const Square bishop,
                           const Bitboard& occupiedRot45Right,
                           const Bitboard& occupiedRot45Left,
                           Bitboard* outAttacksBitboards) {
  if (bishop == Square::NONE) {
    outAttacksBitboards[0] = Bitboard();
    return;
  }
  outAttacksBitboards[0] = diagRightAttacks[getDiagRightBlockPattern(
                                    occupiedRot45Right, bishop)][bishop] |
                                    diagLeftAttacks[getDiagLeftBlockPattern(
                                        occupiedRot45Left, bishop)][bishop];
}
void rookAttackBitboards(const Square rook,
                         const Bitboard& occupied,
                         const Bitboard& occupiedRot90,
                         Bitboard* outAttacksBitboards) {
  if (rook == Square::NONE) {
    outAttacksBitboards[0] = Bitboard();
    return;
  }
  outAttacksBitboards[0] = rankAttacks[getRankBlockPattern(occupied, rook)][rook] |
                 fileAttacks[getFileBlockPattern(occupiedRot90, rook)][rook];
}
void horseAttackBitboards(const Square horse,
                          const Bitboard& occupiedRot45Right,
                          const Bitboard& occupiedRot45Left,
                          Bitboard* outAttacksBitboards) {
  if (horse == Square::NONE) {
    outAttacksBitboards[0] = Bitboard();
    return;
  }
  Bitboard horseBB = Bitboard(horse);
  outAttacksBitboards[0] = diagRightAttacks[getDiagRightBlockPattern(
                                   occupiedRot45Right, horse)][horse] |
                                   diagLeftAttacks[getDiagLeftBlockPattern(
                                       occupiedRot45Left, horse)][horse] |
                                   moveN(horseBB) | moveE(horseBB) |
                                   moveS(horseBB) | moveW(horseBB);
  ;
}
void dragonAttackBitboards(const Square dragon,
                           const Bitboard& occupied,
                           const Bitboard& occupiedRot90,
                           Bitboard* outAttacksBitboards) {
  if (dragon == Square::NONE) {
    outAttacksBitboards[0] = Bitboard();
    return;
  }
  Bitboard dragonBB = Bitboard(dragon);
  outAttacksBitboards[0] = rankAttacks[getRankBlockPattern(occupied, dragon)][dragon] |
                 fileAttacks[getFileBlockPattern(occupiedRot90, dragon)]
                            [dragon] |
                 moveNW(dragonBB) | moveNE(dragonBB) | moveSE(dragonBB) |
                 moveSW(dragonBB);
  ;
  ;
}

size_t countWhitePawnsMoves(const Bitboard pawns, const Bitboard& validMoves) {
  Bitboard attacks;
  whitePawnsAttackBitboards(pawns, &attacks);
  attacks &= validMoves;
  return std::popcount<uint32_t>(attacks[TOP]) +
         std::popcount<uint32_t>(attacks[MID]) +
         std::popcount<uint32_t>(attacks[BOTTOM] & (~BOTTOM_RANK)) * 2 +
         std::popcount<uint32_t>(attacks[BOTTOM] & BOTTOM_RANK);
}

size_t countBlackPawnsMoves(const Bitboard pawns, const Bitboard& validMoves) {
  Bitboard attacks;
  blackPawnsAttackBitboards(pawns, &attacks);
  attacks &= validMoves;
  return std::popcount<uint32_t>(attacks[BOTTOM]) +
         std::popcount<uint32_t>(attacks[MID]) +
         std::popcount<uint32_t>(attacks[TOP] & (~TOP_RANK)) * 2 +
         std::popcount<uint32_t>(attacks[TOP] & TOP_RANK);
}

size_t countWhiteKnightsMoves(const Bitboard knights,
                              const Bitboard& validMoves) {
  Bitboard attacks[2];
  whiteKnightsAttackBitboards(knights, attacks);
  attacks[0] &= validMoves;
  attacks[1] &= validMoves;
  return std::popcount<uint32_t>(attacks[0][TOP]) +
         std::popcount<uint32_t>(attacks[0][MID]) +
         std::popcount<uint32_t>(attacks[0][BOTTOM] & (TOP_RANK)) * 2 +
         std::popcount<uint32_t>(attacks[0][BOTTOM] & (~TOP_RANK)) +
         std::popcount<uint32_t>(attacks[1][TOP]) +
         std::popcount<uint32_t>(attacks[1][MID]) +
         std::popcount<uint32_t>(attacks[1][BOTTOM] & (TOP_RANK)) * 2 +
         std::popcount<uint32_t>(attacks[1][BOTTOM] & (~TOP_RANK));
}

size_t countBlackKnightsMoves(const Bitboard knights,
                              const Bitboard& validMoves) {
  Bitboard attacks[2];
  blackKnightsAttackBitboards(knights, attacks);
  attacks[0] &= validMoves;
  attacks[1] &= validMoves;
  return std::popcount<uint32_t>(attacks[0][BOTTOM]) +
                  std::popcount<uint32_t>(attacks[0][MID]) +
                  std::popcount<uint32_t>(attacks[0][TOP] & (BOTTOM_RANK)) * 2 +
                  std::popcount<uint32_t>(attacks[0][TOP] & (~BOTTOM_RANK)) +
                  std::popcount<uint32_t>(attacks[1][BOTTOM]) +
                  std::popcount<uint32_t>(attacks[1][MID]) +
                  std::popcount<uint32_t>(attacks[1][TOP] & (BOTTOM_RANK)) * 2 +
                  std::popcount<uint32_t>(attacks[1][TOP] & (~BOTTOM_RANK));
}

size_t countWhiteSilverGeneralsMoves(const Bitboard silverGenerals,
                                     const Bitboard& validMoves) {
  Bitboard attacks[5];
  whiteSilverGeneralsAttackBitboards(silverGenerals, attacks);
  attacks[0] &= validMoves;
  attacks[1] &= validMoves;
  attacks[2] &= validMoves;
  attacks[3] &= validMoves;
  attacks[4] &= validMoves;
  return std::popcount<uint32_t>(attacks[0][TOP]) +
         std::popcount<uint32_t>(attacks[0][MID]) +
         std::popcount<uint32_t>(attacks[0][BOTTOM]) * 2 +
         std::popcount<uint32_t>(attacks[1][TOP]) +
         std::popcount<uint32_t>(attacks[1][MID]) +
         std::popcount<uint32_t>(attacks[1][BOTTOM]) * 2 +
         std::popcount<uint32_t>(attacks[2][TOP]) +
         std::popcount<uint32_t>(attacks[2][MID]) +
         std::popcount<uint32_t>(attacks[2][BOTTOM]) * 2 +
         std::popcount<uint32_t>(attacks[3][TOP]) +
         std::popcount<uint32_t>(attacks[3][MID]) +
         std::popcount<uint32_t>(attacks[3][BOTTOM]) * 2 +
         std::popcount<uint32_t>(attacks[4][TOP]) +
         std::popcount<uint32_t>(attacks[4][MID]) +
         std::popcount<uint32_t>(attacks[4][BOTTOM]) * 2;
}

size_t countBlackSilverGeneralsMoves(const Bitboard silverGenerals,
                                     const Bitboard& validMoves) {
  Bitboard attacks[5];
  blackSilverGeneralsAttackBitboards(silverGenerals, attacks);
  attacks[0] &= validMoves;
  attacks[1] &= validMoves;
  attacks[2] &= validMoves;
  attacks[3] &= validMoves;
  attacks[4] &= validMoves;
  return std::popcount<uint32_t>(attacks[0][BOTTOM]) +
         std::popcount<uint32_t>(attacks[0][MID]) +
         std::popcount<uint32_t>(attacks[0][TOP]) * 2 +
         std::popcount<uint32_t>(attacks[1][BOTTOM]) +
         std::popcount<uint32_t>(attacks[1][MID]) +
         std::popcount<uint32_t>(attacks[1][TOP]) * 2 +
         std::popcount<uint32_t>(attacks[2][BOTTOM]) +
         std::popcount<uint32_t>(attacks[2][MID]) +
         std::popcount<uint32_t>(attacks[2][TOP]) * 2 +
         std::popcount<uint32_t>(attacks[3][BOTTOM]) +
         std::popcount<uint32_t>(attacks[3][MID]) +
         std::popcount<uint32_t>(attacks[3][TOP]) * 2 +
         std::popcount<uint32_t>(attacks[4][BOTTOM]) +
         std::popcount<uint32_t>(attacks[4][MID]) +
         std::popcount<uint32_t>(attacks[4][TOP]) * 2;
}

size_t countWhiteGoldGeneralsMoves(const Bitboard goldGenerals,
                                   const Bitboard& validMoves) {
  Bitboard attacks[6];
  whiteGoldGeneralsAttackBitboards(goldGenerals, attacks);
  attacks[0] &= validMoves;
  attacks[1] &= validMoves;
  attacks[2] &= validMoves;
  attacks[3] &= validMoves;
  attacks[4] &= validMoves;
  attacks[5] &= validMoves;
  return std::popcount<uint32_t>(attacks[0][TOP]) +
         std::popcount<uint32_t>(attacks[0][MID]) +
         std::popcount<uint32_t>(attacks[0][BOTTOM]) +
         std::popcount<uint32_t>(attacks[1][TOP]) +
         std::popcount<uint32_t>(attacks[1][MID]) +
         std::popcount<uint32_t>(attacks[1][BOTTOM]) +
         std::popcount<uint32_t>(attacks[2][TOP]) +
         std::popcount<uint32_t>(attacks[2][MID]) +
         std::popcount<uint32_t>(attacks[2][BOTTOM]) +
         std::popcount<uint32_t>(attacks[3][TOP]) +
         std::popcount<uint32_t>(attacks[3][MID]) +
         std::popcount<uint32_t>(attacks[3][BOTTOM]) +
         std::popcount<uint32_t>(attacks[4][TOP]) +
         std::popcount<uint32_t>(attacks[4][MID]) +
         std::popcount<uint32_t>(attacks[4][BOTTOM]) +
         std::popcount<uint32_t>(attacks[5][TOP]) +
         std::popcount<uint32_t>(attacks[5][MID]) +
         std::popcount<uint32_t>(attacks[5][BOTTOM]);
}

size_t countBlackGoldGeneralsMoves(const Bitboard goldGenerals,
                                   const Bitboard& validMoves) {
  Bitboard attacks[6];
  blackGoldGeneralsAttackBitboards(goldGenerals, attacks);
  attacks[0] &= validMoves;
  attacks[1] &= validMoves;
  attacks[2] &= validMoves;
  attacks[3] &= validMoves;
  attacks[4] &= validMoves;
  attacks[5] &= validMoves;
  return std::popcount<uint32_t>(attacks[0][TOP]) +
         std::popcount<uint32_t>(attacks[0][MID]) +
         std::popcount<uint32_t>(attacks[0][BOTTOM]) +
         std::popcount<uint32_t>(attacks[1][TOP]) +
         std::popcount<uint32_t>(attacks[1][MID]) +
         std::popcount<uint32_t>(attacks[1][BOTTOM]) +
         std::popcount<uint32_t>(attacks[2][TOP]) +
         std::popcount<uint32_t>(attacks[2][MID]) +
         std::popcount<uint32_t>(attacks[2][BOTTOM]) +
         std::popcount<uint32_t>(attacks[3][TOP]) +
         std::popcount<uint32_t>(attacks[3][MID]) +
         std::popcount<uint32_t>(attacks[3][BOTTOM]) +
         std::popcount<uint32_t>(attacks[4][TOP]) +
         std::popcount<uint32_t>(attacks[4][MID]) +
         std::popcount<uint32_t>(attacks[4][BOTTOM]) +
         std::popcount<uint32_t>(attacks[5][TOP]) +
         std::popcount<uint32_t>(attacks[5][MID]) +
         std::popcount<uint32_t>(attacks[5][BOTTOM]);
}

size_t countKingMoves(const Square king, const Bitboard& validMoves) {
  Bitboard attacks[8];
  kingAttackBitboards(king, attacks);
  attacks[0] &= validMoves;
  attacks[1] &= validMoves;
  attacks[2] &= validMoves;
  attacks[3] &= validMoves;
  attacks[4] &= validMoves;
  attacks[5] &= validMoves;
  attacks[6] &= validMoves;
  attacks[7] &= validMoves;
  return std::popcount<uint32_t>(attacks[0][TOP]) +
         std::popcount<uint32_t>(attacks[0][MID]) +
         std::popcount<uint32_t>(attacks[0][BOTTOM]) +
         std::popcount<uint32_t>(attacks[1][TOP]) +
         std::popcount<uint32_t>(attacks[1][MID]) +
         std::popcount<uint32_t>(attacks[1][BOTTOM]) +
         std::popcount<uint32_t>(attacks[2][TOP]) +
         std::popcount<uint32_t>(attacks[2][MID]) +
         std::popcount<uint32_t>(attacks[2][BOTTOM]) +
         std::popcount<uint32_t>(attacks[3][TOP]) +
         std::popcount<uint32_t>(attacks[3][MID]) +
         std::popcount<uint32_t>(attacks[3][BOTTOM]) +
         std::popcount<uint32_t>(attacks[4][TOP]) +
         std::popcount<uint32_t>(attacks[4][MID]) +
         std::popcount<uint32_t>(attacks[4][BOTTOM]) +
         std::popcount<uint32_t>(attacks[5][TOP]) +
         std::popcount<uint32_t>(attacks[5][MID]) +
         std::popcount<uint32_t>(attacks[5][BOTTOM]) +
         std::popcount<uint32_t>(attacks[6][TOP]) +
         std::popcount<uint32_t>(attacks[6][MID]) +
         std::popcount<uint32_t>(attacks[6][BOTTOM]) +
         std::popcount<uint32_t>(attacks[7][TOP]) +
         std::popcount<uint32_t>(attacks[7][MID]) +
         std::popcount<uint32_t>(attacks[7][BOTTOM]);
}

size_t countWhiteLancesMoves(const Square lance1,
                             const Square lance2,
                             const Bitboard& validMoves,
                             const Bitboard& occupiedRot90) {
  Bitboard attacks[2];
  whiteLanceAttackBitboards(lance1, occupiedRot90, attacks);
  whiteLanceAttackBitboards(lance2, occupiedRot90, attacks + 1);
  attacks[0] &= validMoves;
  attacks[1] &= validMoves;
  return std::popcount<uint32_t>(attacks[0][TOP]) +
         std::popcount<uint32_t>(attacks[0][MID]) +
         std::popcount<uint32_t>(attacks[0][BOTTOM] & (~BOTTOM_RANK)) * 2 +
         std::popcount<uint32_t>(attacks[0][BOTTOM] & BOTTOM_RANK) +
         std::popcount<uint32_t>(attacks[1][TOP]) +
         std::popcount<uint32_t>(attacks[1][MID]) +
         std::popcount<uint32_t>(attacks[1][BOTTOM] & (~BOTTOM_RANK)) * 2 +
         std::popcount<uint32_t>(attacks[1][BOTTOM] & BOTTOM_RANK);
}

size_t countBlackLancesMoves(const Square lance1,
                             const Square lance2,
                             const Bitboard& validMoves,
                             const Bitboard& occupiedRot90) {
  Bitboard attacks[2];
  blackLanceAttackBitboards(lance1, occupiedRot90, attacks);
  blackLanceAttackBitboards(lance2, occupiedRot90, attacks + 1);
  attacks[0] &= validMoves;
  attacks[1] &= validMoves;
  return std::popcount<uint32_t>(attacks[0][BOTTOM]) +
         std::popcount<uint32_t>(attacks[0][MID]) +
         std::popcount<uint32_t>(attacks[0][TOP] & (~TOP_RANK)) * 2 +
         std::popcount<uint32_t>(attacks[0][TOP] & TOP_RANK) +
         std::popcount<uint32_t>(attacks[1][BOTTOM]) +
         std::popcount<uint32_t>(attacks[1][MID]) +
         std::popcount<uint32_t>(attacks[1][TOP] & (~TOP_RANK)) * 2 +
         std::popcount<uint32_t>(attacks[1][TOP] & TOP_RANK);
}

size_t countWhiteBishopMoves(const Square bishop,
                             const Bitboard& validMoves,
                             const Bitboard& occupiedRot45Right,
                             const Bitboard& occupiedRot45Left) {
  Bitboard attacks;
  bishopAttackBitboards(bishop, occupiedRot45Right, occupiedRot45Left,
                        &attacks);
  attacks &= validMoves;
  int multiplier = squareToRank(bishop) > 5 ? 2 : 1;
  return (std::popcount<uint32_t>(attacks[TOP]) +
          std::popcount<uint32_t>(attacks[MID])) *
             multiplier +
         std::popcount<uint32_t>(attacks[BOTTOM]) * 2;
}

size_t countBlackBishopMoves(const Square bishop,
                             const Bitboard& validMoves,
                             const Bitboard& occupiedRot45Right,
                             const Bitboard& occupiedRot45Left) {
  Bitboard attacks;
  bishopAttackBitboards(bishop, occupiedRot45Right, occupiedRot45Left,
                        &attacks);
  attacks &= validMoves;
  int multiplier = squareToRank(bishop) < 3 ? 2 : 1;
  return (std::popcount<uint32_t>(attacks[BOTTOM]) +
          std::popcount<uint32_t>(attacks[MID])) *
             multiplier +
         std::popcount<uint32_t>(attacks[TOP]) * 2;
}

size_t countWhiteRookMoves(const Square rook,
                           const Bitboard& validMoves,
                           const Bitboard& occupied,
                           const Bitboard& occupiedRot90) {
  Bitboard attacks;
  rookAttackBitboards(rook, occupied, occupiedRot90, &attacks);
  attacks &= validMoves;
  int multiplier = squareToRank(rook) > 5 ? 2 : 1;
  return (std::popcount<uint32_t>(attacks[TOP]) +
          std::popcount<uint32_t>(attacks[MID])) *
             multiplier +
         std::popcount<uint32_t>(attacks[BOTTOM]) * 2;
}

size_t countBlackRookMoves(const Square rook,
                           const Bitboard& validMoves,
                           const Bitboard& occupied,
                           const Bitboard& occupiedRot90) {
  Bitboard attacks;
  rookAttackBitboards(rook, occupied, occupiedRot90, &attacks);
  attacks &= validMoves;
  int multiplier = squareToRank(rook) < 3 ? 2 : 1;
  return (std::popcount<uint32_t>(attacks[BOTTOM]) +
          std::popcount<uint32_t>(attacks[MID])) *
             multiplier +
         std::popcount<uint32_t>(attacks[TOP]) * 2;
}

size_t countHorseMoves(const Square horse,
                       const Bitboard& validMoves,
                       const Bitboard& occupiedRot45Right,
                       const Bitboard& occupiedRot45Left) {
  Bitboard attacks;
  horseAttackBitboards(horse, occupiedRot45Right, occupiedRot45Left, &attacks);
  attacks &= validMoves;
  return std::popcount<uint32_t>(attacks[TOP]) +
         std::popcount<uint32_t>(attacks[MID]) +
         std::popcount<uint32_t>(attacks[BOTTOM]);
}

size_t countDragonMoves(const Square dragon,
                        const Bitboard& validMoves,
                        const Bitboard& occupied,
                        const Bitboard& occupiedRot90) {
  Bitboard attacks;
  dragonAttackBitboards(dragon, occupied, occupiedRot90, &attacks);
  attacks &= validMoves;
  return std::popcount<uint32_t>(attacks[TOP]) +
         std::popcount<uint32_t>(attacks[MID]) +
         std::popcount<uint32_t>(attacks[BOTTOM]);
}

size_t countDropMoves(const PlayerInHandPieces& inHand,
                      const Bitboard& freeSquares,
                      const Bitboard ownPawns,
                      const Bitboard enemyKing,
                      bool isWhite) {
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