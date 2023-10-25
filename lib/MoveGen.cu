#include <algorithm>
#include "MoveGen.h"

std::array<Bitboard, BOARD_SIZE - BOARD_DIM> WhitePawnAttacks =
    initWhitePawnAttacks();
std::array<Bitboard, BOARD_SIZE - 2 * BOARD_DIM> WhiteKnightAttacks =
    initWhiteKnightAttacks();
std::array<Bitboard, BOARD_SIZE> WhiteSilverGeneralAttacks =
    initWhiteSilverGeneralAttacks();
std::array<Bitboard, BOARD_SIZE> WhiteGoldGeneralAttacks =
    initWhiteGoldGeneralAttacks();
std::array<Bitboard, BOARD_SIZE - BOARD_DIM> BlackPawnAttacks =
    initBlackPawnAttacks();
std::array<Bitboard, BOARD_SIZE - 2 * BOARD_DIM> BlackKnightAttacks =
    initBlackKnightAttacks();
std::array<Bitboard, BOARD_SIZE> BlackSilverGeneralAttacks =
    initBlackSilverGeneralAttacks();
std::array<Bitboard, BOARD_SIZE> BlackGoldGeneralAttacks =
    initBlackGoldGeneralAttacks();
std::array<Bitboard, BOARD_SIZE> KingAttacks = initKingAttacks();

std::array<std::array<Bitboard, BOARD_SIZE>, 128> RankAttacks =
    initRankAttacks();
std::array<std::array<Bitboard, BOARD_SIZE>, 128> FileAttacks =
    initFileAttacks();
std::array<std::array<Bitboard, BOARD_SIZE>, 128> DiagRightAttacks =
    initDiagRightAttacks();
std::array<std::array<Bitboard, BOARD_SIZE>, 128> DiagLeftAttacks =
    initDiagLeftAttacks();

std::array<Bitboard, BOARD_SIZE - BOARD_DIM> initWhitePawnAttacks() {
  // Without last row because forced promotion
  std::array<Bitboard, BOARD_SIZE - BOARD_DIM> result;
  std::array<short, BOARD_SIZE> mat;

  for (int i = 0; i < result.size(); i++) {
    mat.fill(0);
    mat[i + BOARD_DIM] = 1;  // 1 down
    result[i] = Bitboard(mat);
  }

  return result;
}

std::array<Bitboard, BOARD_SIZE - BOARD_DIM> initBlackPawnAttacks() {
  // Without last row because forced promotion
  std::array<Bitboard, BOARD_SIZE - BOARD_DIM> result;
  std::array<short, BOARD_SIZE> mat;

  for (int i = result.size() - 1; i >= BOARD_DIM; i--) {
    mat.fill(0);
    mat[i - BOARD_DIM] = 1;  // 1 up
    result[i] = Bitboard(mat);
  }

  return result;
}

std::array<Bitboard, BOARD_SIZE - 2 * BOARD_DIM> initWhiteKnightAttacks() {
  // Without two last rows because forced promotion
  std::array<Bitboard, BOARD_SIZE - 2 * BOARD_DIM> result;
  std::array<short, BOARD_SIZE> mat;

  for (int i = 0; i < result.size(); i++) {
    mat.fill(0);
    if (i % BOARD_DIM != 0)
      mat[i + 2 * BOARD_DIM - 1] = 1;  // 2 down 1 left
    if ((i + 1) % BOARD_DIM != 0)
      mat[i + 2 * BOARD_DIM + 1] = 1;  // 2 down 1 right
    result[i] = Bitboard(mat);
  }

  return result;
}

std::array<Bitboard, BOARD_SIZE - 2 * BOARD_DIM> initBlackKnightAttacks() {
  // Without two last rows because forced promotion
  std::array<Bitboard, BOARD_SIZE - 2 * BOARD_DIM> result;
  std::array<short, BOARD_SIZE> mat;

  for (int i = result.size() - 1; i >= 2 * BOARD_DIM; i--) {
    mat.fill(0);
    if (i % BOARD_DIM != 0)
      mat[i - 2 * BOARD_DIM - 1] = 1;  // 2 up 1 left
    if ((i + 1) % BOARD_DIM != 0)
      mat[i - 2 * BOARD_DIM + 1] = 1;  // 2 up 1 right
    result[i] = Bitboard(mat);
  }

  return result;
}

std::array<Bitboard, BOARD_SIZE> initWhiteSilverGeneralAttacks() {
  std::array<Bitboard, BOARD_SIZE> result;
  std::array<short, BOARD_SIZE> mat;

  for (int i = 0; i < result.size(); i++) {
    mat.fill(0);
    if (i < BOARD_SIZE - BOARD_DIM) {  // forward moves
      mat[i + BOARD_DIM] = 1;          // 1 down
      if (i % BOARD_DIM != 0)
        mat[i + BOARD_DIM - 1] = 1;  // 1 down 1 left
      if ((i + 1) % BOARD_DIM != 0)
        mat[i + BOARD_DIM + 1] = 1;  // 1 down 1 right
    }
    if (i >= BOARD_DIM) {  // backwords moves
      if (i % BOARD_DIM != 0)
        mat[i - BOARD_DIM - 1] = 1;  // 1 up 1 left
      if ((i + 1) % BOARD_DIM != 0)
        mat[i - BOARD_DIM + 1] = 1;  // 1 up 1 right
    }
    result[i] = Bitboard(mat);
  }

  return result;
}

std::array<Bitboard, BOARD_SIZE> initBlackSilverGeneralAttacks() {
  std::array<Bitboard, BOARD_SIZE> result;
  std::array<short, BOARD_SIZE> mat;

  for (int i = 0; i < result.size(); i++) {
    mat.fill(0);
    if (i >= BOARD_DIM) {      // forward moves
      mat[i - BOARD_DIM] = 1;  // 1 up
      if (i % BOARD_DIM != 0)
        mat[i - BOARD_DIM - 1] = 1;  // 1 up 1 left
      if ((i + 1) % BOARD_DIM != 0)
        mat[i - BOARD_DIM + 1] = 1;  // 1 up 1 right
    }
    if (i < BOARD_SIZE - BOARD_DIM) {  // backwords moves
      if (i % BOARD_DIM != 0)
        mat[i + BOARD_DIM - 1] = 1;  // 1 down 1 left
      if ((i + 1) % BOARD_DIM != 0)
        mat[i + BOARD_DIM + 1] = 1;  // 1 down 1 right
    }
    result[i] = Bitboard(mat);
  }

  return result;
}
std::array<Bitboard, BOARD_SIZE> initWhiteGoldGeneralAttacks() {
  std::array<Bitboard, BOARD_SIZE> result;
  std::array<short, BOARD_SIZE> mat;

  for (int i = 0; i < result.size(); i++) {
    mat.fill(0);
    if (i < BOARD_SIZE - BOARD_DIM) {  // forward moves
      mat[i + BOARD_DIM] = 1;          // 1 down
      if (i % BOARD_DIM != 0)
        mat[i + BOARD_DIM - 1] = 1;  // 1 down 1 left
      if ((i + 1) % BOARD_DIM != 0)
        mat[i + BOARD_DIM + 1] = 1;  // 1 down 1 right
    }
    // side moves
    if (i % BOARD_DIM != 0)
      mat[i - 1] = 1;  // 1 left
    if ((i + 1) % BOARD_DIM != 0)
      mat[i + 1] = 1;          // 1 right
    if (i >= BOARD_DIM) {      // backwords moves
      mat[i - BOARD_DIM] = 1;  // 1 up
    }
    result[i] = Bitboard(mat);
  }

  return result;
}

std::array<Bitboard, BOARD_SIZE> initBlackGoldGeneralAttacks() {
  std::array<Bitboard, BOARD_SIZE> result;
  std::array<short, BOARD_SIZE> mat;

  for (int i = 0; i < result.size(); i++) {
    mat.fill(0);
    if (i >= BOARD_DIM) {      // forward moves
      mat[i - BOARD_DIM] = 1;  // 1 up
      if (i % BOARD_DIM != 0)
        mat[i - BOARD_DIM - 1] = 1;  // 1 up 1 left
      if ((i + 1) % BOARD_DIM != 0)
        mat[i - BOARD_DIM + 1] = 1;  // 1 up 1 right
    }
    // side moves
    if (i % BOARD_DIM != 0)
      mat[i - 1] = 1;  // 1 left
    if ((i + 1) % BOARD_DIM != 0)
      mat[i + 1] = 1;                  // 1 right
    if (i < BOARD_SIZE - BOARD_DIM) {  // backwords moves
      mat[i + BOARD_DIM] = 1;          // 1 down
    }
    result[i] = Bitboard(mat);
  }

  return result;
}
std::array<Bitboard, BOARD_SIZE> initKingAttacks() {
  std::array<Bitboard, BOARD_SIZE> result;
  std::array<short, BOARD_SIZE> mat;

  for (int i = 0; i < result.size(); i++) {
    mat.fill(0);
    if (i < BOARD_SIZE - BOARD_DIM) {
      mat[i + BOARD_DIM] = 1;  // 1 down
      if (i % BOARD_DIM != 0)
        mat[i + BOARD_DIM - 1] = 1;  // 1 down 1 left
      if ((i + 1) % BOARD_DIM != 0)
        mat[i + BOARD_DIM + 1] = 1;  // 1 down 1 right
    }
    if (i % BOARD_DIM != 0)
      mat[i - 1] = 1;  // 1 left
    if ((i + 1) % BOARD_DIM != 0)
      mat[i + 1] = 1;  // 1 right
    if (i >= BOARD_DIM) {
      mat[i - BOARD_DIM] = 1;  // 1 up
      if (i % BOARD_DIM != 0)
        mat[i - BOARD_DIM - 1] = 1;  // 1 up 1 left
      if ((i + 1) % BOARD_DIM != 0)
        mat[i - BOARD_DIM + 1] = 1;  // 1 up 1 right
    }
    result[i] = Bitboard(mat);
  }

  return result;
}

#define BIT_SET(var, pos) ((var) & (1 << (pos)))

int getRankBlockPattern(Bitboard& bb, int fieldIdx) {
  uint32_t& region = bb[fieldIdx / REGION_SIZE];
  int rowsBeforeInRegion = (fieldIdx / BOARD_DIM) % 3;
  uint32_t result = region << 5 << (rowsBeforeInRegion * BOARD_DIM) << 1 >> 25;
  return result;
}

std::array<short, 9> blockPatternToRow(int blockPattern) {
  int block = blockPattern << 1;
  std::array<short, 9> result;
  result.fill(0);
  int mask = 1 << 7;
  for (int i = 1; i < 8; i++) {
    if (block & mask)
      result[i] = 1;
    mask = mask >> 1;
  }
  return result;
}

std::array<std::array<Bitboard, BOARD_SIZE>, 128> initRankAttacks() {
  std::array<std::array<Bitboard, BOARD_SIZE>, 128> result;
  std::array<short, BOARD_SIZE> mat;

  for (int i = 0; i < BOARD_SIZE; i++) {
    int columnIdx = i % BOARD_DIM;
    int rowIdx = i / BOARD_DIM;
    for (int blockPattern = 0; blockPattern < 128; blockPattern++) {
      mat.fill(0);

      std::array<short, BOARD_DIM> blockRow = blockPatternToRow(blockPattern);
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
      // Fill first and last, becuase this moves are always valid
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

int getFileBlockPattern(Bitboard& bbRot90, int fieldIdx) {
  uint32_t& region = bbRot90[fieldIdx / REGION_SIZE];
  int rowsBeforeInRegion = (fieldIdx / BOARD_DIM) % 3;
  return region << (rowsBeforeInRegion * BOARD_DIM) << 1 >> 25;
}

std::array<std::array<Bitboard, BOARD_SIZE>, 128> initFileAttacks() {
  std::array<std::array<Bitboard, BOARD_SIZE>, 128> result;

  for (int i = 0; i < BOARD_SIZE; i++) {
    for (int blockPattern = 0; blockPattern < 128; blockPattern++) {
      result[blockPattern][Rotate90AntiClockwise(i)] =
          Rotate90Clockwise(RankAttacks[blockPattern][i]);
    }
  }

  return result;
}

int getDiagRightBlockPattern(Bitboard& bbRot45Right, int fieldIdx) {
  static const int regionIdx[BOARD_SIZE] = {
      1, 0, 0, 0, 0, 0, 0, 1, 1,  //
      0, 0, 0, 0, 0, 0, 1, 1, 1,  //
      0, 0, 0, 0, 0, 1, 1, 1, 2,  //
      0, 0, 0, 0, 1, 1, 1, 2, 2,  //
      0, 0, 0, 1, 1, 1, 2, 2, 2,  //
      0, 0, 1, 1, 1, 2, 2, 2, 2,  //
      0, 1, 1, 1, 2, 2, 2, 2, 2,  //
      1, 1, 1, 2, 2, 2, 2, 2, 2,  //
      1, 1, 2, 2, 2, 2, 2, 2, 1,
  };

  static const int shiftRight[BOARD_SIZE] = {
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

  static const int mask[BOARD_SIZE] = {
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

  uint32_t& region = bbRot45Right[regionIdx[fieldIdx]];
  int aftershift = region >> shiftRight[fieldIdx] >> 1;
  int value = aftershift & (mask[fieldIdx] / 4);
  return value;
}

std::array<std::array<Bitboard, BOARD_SIZE>, 128> initDiagRightAttacks() {
  std::array<std::array<Bitboard, BOARD_SIZE>, 128> result;
  std::array<short, BOARD_SIZE> mat;

  for (int i = 0; i < BOARD_SIZE; i++) {
    int columnIdx = i % BOARD_DIM;
    int rowIdx = i / BOARD_DIM;
    int diagLength = (columnIdx + rowIdx < BOARD_DIM)
                         ? columnIdx + rowIdx + 1
                         : (BOARD_DIM - columnIdx) + (BOARD_DIM - rowIdx) - 1;
    int maxPattern = 1 << std::max((diagLength - 2), 0);
    for (int blockPattern = 0; blockPattern < maxPattern; blockPattern++) {
      mat.fill(0);
      std::array<short, BOARD_DIM> blockRow =
          blockPatternToRow(blockPattern << (BOARD_DIM - diagLength));
      int tmpColIdx = columnIdx;
      int tmpRowIdx = rowIdx;
      int idxInDiag = 0;
      while (tmpRowIdx < BOARD_DIM - 1 && tmpColIdx > 0) {
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

int getDiagLeftBlockPattern(Bitboard& bbRot45Left, int fieldIdx) {
  static const int regionIdx[BOARD_SIZE] = {
      1, 1, 0, 0, 0, 0, 0, 0, 1,  //
      1, 1, 1, 0, 0, 0, 0, 0, 0,  //
      2, 1, 1, 1, 0, 0, 0, 0, 0,  //
      2, 2, 1, 1, 1, 0, 1, 0, 0,  //
      2, 2, 2, 1, 1, 1, 0, 0, 0,  //
      2, 2, 2, 2, 1, 1, 1, 0, 0,  //
      2, 2, 2, 2, 2, 1, 1, 1, 0,  //
      2, 2, 2, 2, 2, 2, 1, 1, 1,  //
      1, 2, 2, 2, 2, 2, 2, 1, 1,
  };

  static const int shiftRight[BOARD_SIZE] = {
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

  static const int mask[BOARD_SIZE] = {
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

  uint32_t& region = bbRot45Left[regionIdx[fieldIdx]];
  int aftershift = region >> shiftRight[fieldIdx] >> 1;
  int value = aftershift & (mask[fieldIdx] / 4);
  return value;
}

std::array<std::array<Bitboard, BOARD_SIZE>, 128> initDiagLeftAttacks() {
  std::array<std::array<Bitboard, BOARD_SIZE>, 128> result;
  std::array<short, BOARD_SIZE> mat;

  for (int i = 0; i < BOARD_SIZE; i++) {
    int columnIdx = i % BOARD_DIM;
    int rowIdx = i / BOARD_DIM;
    int diagLength = ((BOARD_DIM - columnIdx) + rowIdx < BOARD_DIM)
                         ? (BOARD_DIM - columnIdx) + rowIdx
                         : columnIdx + (BOARD_DIM - rowIdx);
    int maxPattern = 1 << std::max((diagLength - 2), 0);
    for (int blockPattern = 0; blockPattern < maxPattern; blockPattern++) {
      mat.fill(0);
      std::array<short, BOARD_DIM> blockRow =
          blockPatternToRow(blockPattern << (BOARD_DIM - diagLength));
      int tmpColIdx = columnIdx;
      int tmpRowIdx = rowIdx;
      int idxInDiag = 0;
      while (tmpRowIdx >0 && tmpColIdx > 0) {
        tmpRowIdx--;
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