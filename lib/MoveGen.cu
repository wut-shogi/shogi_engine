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
  std::array<bool, BOARD_SIZE> mat;

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
  std::array<bool, BOARD_SIZE> mat;

  for (int i = 0; i < result.size(); i++) {
    mat.fill(0);
    mat[i - BOARD_DIM] = 1;  // 1 up
    result[i] = Bitboard(mat);
  }

  return result;
}

std::array<Bitboard, BOARD_SIZE - 2 * BOARD_DIM> initWhiteKnightAttacks() {
  // Without two last rows because forced promotion
  std::array<Bitboard, BOARD_SIZE - 2 * BOARD_DIM> result;
  std::array<bool, BOARD_SIZE> mat;

  for (int i = 0; i < result.size(); i++) {
    mat.fill(0);
    int idx;
    if (i % BOARD_DIM != 0)
      mat[i + 2 * BOARD_DIM - 1] = 1;  // 2 down 1 left
    if (i % (BOARD_DIM - 1) != 0)
      mat[i + 2 * BOARD_DIM + 1] = 1;  // 2 down 1 right
    result[i] = Bitboard(mat);
  }

  return result;
}

std::array<Bitboard, BOARD_SIZE - 2 * BOARD_DIM> initBlackKnightAttacks() {
  // Without two last rows because forced promotion
  std::array<Bitboard, BOARD_SIZE - 2 * BOARD_DIM> result;
  std::array<bool, BOARD_SIZE> mat;

  for (int i = 0; i < result.size(); i++) {
    mat.fill(0);
    int idx;
    if (i % BOARD_DIM != 0)
      mat[i - 2 * BOARD_DIM - 1] = 1;  // 2 up 1 left
    if (i % (BOARD_DIM - 1) != 0)
      mat[i - 2 * BOARD_DIM + 1] = 1;  // 2 up 1 right
    result[i] = Bitboard(mat);
  }

  return result;
}

std::array<Bitboard, BOARD_SIZE> initWhiteSilverGeneralAttacks() {
  std::array<Bitboard, BOARD_SIZE> result;
  std::array<bool, BOARD_SIZE> mat;

  for (int i = 0; i < result.size(); i++) {
    mat.fill(0);
    if (i < BOARD_SIZE - BOARD_DIM) {  // forward moves
      mat[i + BOARD_DIM] = 1;          // 1 down
      if (i % BOARD_DIM != 0)
        mat[i + BOARD_DIM - 1] = 1;  // 1 down 1 left
      if (i % (BOARD_DIM - 1) != 0)
        mat[i + BOARD_DIM + 1] = 1;  // 1 down 1 right
    }
    if (i >= BOARD_DIM) {  // backwords moves
      if (i % BOARD_DIM != 0)
        mat[i - BOARD_DIM - 1] = 1;  // 1 up 1 left
      if (i % (BOARD_DIM - 1) != 0)
        mat[i - BOARD_DIM + 1] = 1;  // 1 up 1 right
    }
  }

  return result;
}

std::array<Bitboard, BOARD_SIZE> initBlackSilverGeneralAttacks() {
  std::array<Bitboard, BOARD_SIZE> result;
  std::array<bool, BOARD_SIZE> mat;

  for (int i = 0; i < result.size(); i++) {
    mat.fill(0);
    if (i >= BOARD_DIM) {      // forward moves
      mat[i - BOARD_DIM] = 1;  // 1 up
      if (i % BOARD_DIM != 0)
        mat[i - BOARD_DIM - 1] = 1;  // 1 up 1 left
      if (i % (BOARD_DIM - 1) != 0)
        mat[i - BOARD_DIM + 1] = 1;  // 1 up 1 right
    }
    if (i < BOARD_SIZE - BOARD_DIM) {  // backwords moves
      if (i % BOARD_DIM != 0)
        mat[i + BOARD_DIM - 1] = 1;  // 1 down 1 left
      if (i % (BOARD_DIM - 1) != 0)
        mat[i + BOARD_DIM + 1] = 1;  // 1 down 1 right
    }
  }

  return result;
}
std::array<Bitboard, BOARD_SIZE> initWhiteGoldGeneralAttacks() {
  std::array<Bitboard, BOARD_SIZE> result;
  std::array<bool, BOARD_SIZE> mat;

  for (int i = 0; i < result.size(); i++) {
    mat.fill(0);
    if (i < BOARD_SIZE - BOARD_DIM) {  // forward moves
      mat[i + BOARD_DIM] = 1;          // 1 down
      if (i % BOARD_DIM != 0)
        mat[i + BOARD_DIM - 1] = 1;  // 1 down 1 left
      if (i % (BOARD_DIM - 1) != 0)
        mat[i + BOARD_DIM + 1] = 1;  // 1 down 1 right
    }
    // side moves
    if (i % BOARD_DIM != 0)
      mat[i - 1] = 1;  // 1 left
    if (i % (BOARD_DIM - 1) != 0)
      mat[i + 1] = 1;          // 1 right
    if (i >= BOARD_DIM) {      // backwords moves
      mat[i - BOARD_DIM] = 1;  // 1 up
    }
  }

  return result;
}

std::array<Bitboard, BOARD_SIZE> initBlackGoldGeneralAttacks() {
  std::array<Bitboard, BOARD_SIZE> result;
  std::array<bool, BOARD_SIZE> mat;

  for (int i = 0; i < result.size(); i++) {
    mat.fill(0);
    if (i >= BOARD_DIM) {      // forward moves
      mat[i - BOARD_DIM] = 1;  // 1 up
      if (i % BOARD_DIM != 0)
        mat[i - BOARD_DIM - 1] = 1;  // 1 up 1 left
      if (i % (BOARD_DIM - 1) != 0)
        mat[i - BOARD_DIM + 1] = 1;  // 1 up 1 right
    }
    // side moves
    if (i % BOARD_DIM != 0)
      mat[i - 1] = 1;  // 1 left
    if (i % (BOARD_DIM - 1) != 0)
      mat[i + 1] = 1;                  // 1 right
    if (i < BOARD_SIZE - BOARD_DIM) {  // backwords moves
      mat[i + BOARD_DIM] = 1;          // 1 down
    }
  }

  return result;
}
std::array<Bitboard, BOARD_SIZE> initKingAttacks() {
  std::array<Bitboard, BOARD_SIZE> result;
  std::array<bool, BOARD_SIZE> mat;

  for (int i = 0; i < result.size(); i++) {
    mat.fill(0);
    if (i < BOARD_SIZE - BOARD_DIM) {
      mat[i + BOARD_DIM] = 1;  // 1 down
      if (i % BOARD_DIM != 0)
        mat[i + BOARD_DIM - 1] = 1;  // 1 down 1 left
      if (i % (BOARD_DIM - 1) != 0)
        mat[i + BOARD_DIM + 1] = 1;  // 1 down 1 right
    }
    if (i % BOARD_DIM != 0)
      mat[i - 1] = 1;  // 1 left
    if (i % (BOARD_DIM - 1) != 0)
      mat[i + 1] = 1;  // 1 right
    if (i >= BOARD_DIM) {
      mat[i - BOARD_DIM] = 1;  // 1 up
      if (i % BOARD_DIM != 0)
        mat[i - BOARD_DIM - 1] = 1;  // 1 up 1 left
      if (i % (BOARD_DIM - 1) != 0)
        mat[i - BOARD_DIM + 1] = 1;  // 1 up 1 right
    }
  }

  return result;
}

#define BIT_SET(var, pos) ((var) & (1 << (pos)))

int getRankBlockPattern(Bitboard& bb, int fieldIdx) {
  uint32_t& region = bb[fieldIdx / REGION_SIZE];
  int rowsBeforeInRegion = (fieldIdx / BOARD_DIM) % 3;
  return region << (rowsBeforeInRegion * BOARD_DIM) << 1 >> 25;
}

std::array<bool, 9> blockPatternToRow(int blockPattern) {
  std::array<bool, 9> result;
  result.fill(0);
  for (int i = 7; i > 0; i--) {
    result[8 - i] = BIT_SET(blockPattern, i) ? 1 : 0;
  }
  return result;
}

std::array<std::array<Bitboard, BOARD_SIZE>, 128> initRankAttacks() {
  std::array<std::array<Bitboard, BOARD_SIZE>, 128> result;
  std::array<bool, BOARD_SIZE> mat;

  for (int i = 0; i < BOARD_SIZE; i++) {
    int columnIdx = i % BOARD_DIM;
    int rowIdx = i / BOARD_DIM;
    for (int blockPattern = 0; blockPattern < 128; blockPattern++) {
      mat.fill(0);

      std::array<bool, 9> blockRow = blockPatternToRow(blockPattern);
      int lastLeft = 0;
      int firstRight = 0;
      for (int col = 0; col < columnIdx; col++) {
        if (blockRow[i]) {
          lastLeft = i;
        }
      }
      for (int col = columnIdx + 1; col < blockRow.size(); col++) {
        if (blockRow[i]) {
          firstRight = i;
          break;
        }
      }
      // Fill first and last, becuase this moves are always valid
      blockRow.front() = 1;
      blockRow.back() = 1;
      for (int blockIdx = 0; blockIdx < blockRow.size(); blockIdx++) {
        mat[rowIdx * BOARD_DIM + blockIdx] = blockRow[blockIdx];
      }
      result[i][blockPattern] = mat;
    }
  }

  return result;
}

int getFileBlockPattern(Bitboard& bbRot90, int fieldIdx) {
  int rotatedfileIdx = Rotate90(fieldIdx);
  uint32_t& region = bbRot90[rotatedfileIdx / REGION_SIZE];
  int rowsBeforeInRegion = (rotatedfileIdx / BOARD_DIM) % 3;
  return region << (rowsBeforeInRegion * BOARD_DIM) << 1 >> 25;
}

std::array<std::array<Bitboard, BOARD_SIZE>, 128> initFileAttacks() {
  std::array<std::array<Bitboard, BOARD_SIZE>, 128> result;
  std::array<bool, BOARD_SIZE> mat;

  for (int i = 0; i < BOARD_SIZE; i++) {
    int columnIdx = i % BOARD_DIM;
    int rowIdx = i / BOARD_DIM;
    for (int blockPattern = 0; blockPattern < 128; blockPattern++) {
      mat.fill(0);

      std::array<bool, 9> blockRow = blockPatternToRow(blockPattern);
      int lastLeft = 0;
      int firstRight = 0;
      for (int col = 0; col < columnIdx; col++) {
        if (blockRow[i]) {
          lastLeft = i;
        }
      }
      for (int col = columnIdx + 1; col < blockRow.size(); col++) {
        if (blockRow[i]) {
          firstRight = i;
          break;
        }
      }
      // Fill first and last, becuase this moves are always valid
      blockRow.front() = 1;
      blockRow.back() = 1;
      for (int blockIdx = 0; blockIdx < blockRow.size(); blockIdx++) {
        mat[columnIdx * BOARD_DIM + blockIdx] = blockRow[blockIdx];
      }
      result[i][blockPattern] = mat;
    }
  }

  return result;
}


std::array<std::array<Bitboard, BOARD_SIZE>, 128> initDiagRightAttacks() {}

std::array<std::array<Bitboard, BOARD_SIZE>, 128> initDiagLeftAttacks() {}