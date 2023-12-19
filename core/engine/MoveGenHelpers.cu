//#include <algorithm>
//#include "MoveGenHelpers.h"
//
//namespace shogi {
//namespace engine {
//    #define ARRAY_SIZE 10368
///// Static arrays initialization
//// Attack bitboards
//namespace CPU {
//std::array<Bitboard, ARRAY_SIZE> initRankAttacks();
//std::array<Bitboard, ARRAY_SIZE> initFileAttacks();
//std::array<Bitboard, ARRAY_SIZE> initDiagRightAttacks();
//std::array<Bitboard, ARRAY_SIZE> initDiagLeftAttacks();
//std::array<Bitboard, BOARD_DIM> initRankMask();
//std::array<Bitboard, BOARD_DIM> initFileMask();
//
//static std::array<Bitboard, ARRAY_SIZE> rankAttacks =
//    initRankAttacks();
//static std::array<Bitboard, ARRAY_SIZE> fileAttacks =
//    initFileAttacks();
//static std::array<Bitboard, ARRAY_SIZE> diagRightAttacks =
//    initDiagRightAttacks();
//static std::array<Bitboard, ARRAY_SIZE> diagLeftAttacks =
//    initDiagLeftAttacks();
//
//static std::array<Bitboard, BOARD_DIM> rankMask = initRankMask();
//static std::array<Bitboard, BOARD_DIM> fileMask = initFileMask();
//
//
//uint32_t getDiagRightBlockPattern(const Bitboard& occupied, Square square) {
//  static const uint32_t startingSquare[BOARD_SIZE] = {
//      0, 1,  2,  3,  4,  5,  6,  7,  8,   //
//      1, 2,  3,  4,  5,  6,  7,  8,  17,  //
//      2, 3,  4,  5,  6,  7,  8,  17, 26,  //
//      3, 4,  5,  6,  7,  8,  17, 26, 35,  //
//      4, 5,  6,  7,  8,  17, 26, 35, 44,  //
//      5, 6,  7,  8,  17, 26, 35, 44, 53,  //
//      6, 7,  8,  17, 26, 35, 44, 53, 62,  //
//      7, 8,  17, 26, 35, 44, 53, 62, 71,  //
//      8, 17, 26, 35, 44, 53, 62, 71, 80,
//  };
//  uint32_t result = 0;
//  int len = startingSquare[square] >= 9 ? 7 - startingSquare[square] / 9
//                                       : startingSquare[square] - 1;
//  for (int i = 0; i < len; i++) {
//    result += occupied.GetBit(
//                  static_cast<Square>(startingSquare[square] + i * SW + SW))
//              << i;
//  }
//  return result;
//}
//
//uint32_t getDiagLeftBlockPattern(const Bitboard& occupied, Square square) {
//  static const uint32_t startingSquare[BOARD_SIZE] = {
//      0,  1,  2,  3,  4,  5,  6,  7, 8,  //
//      9,  0,  1,  2,  3,  4,  5,  6, 7,  //
//      18, 9,  0,  1,  2,  3,  4,  5, 6,  //
//      27, 18, 9,  0,  1,  2,  3,  4, 5,  //
//      36, 27, 18, 9,  0,  1,  2,  3, 4,  //
//      45, 36, 27, 18, 9,  0,  1,  2, 3,  //
//      54, 45, 36, 27, 18, 9,  0,  1, 2,  //
//      63, 54, 45, 36, 27, 18, 9,  0, 1,  //
//      72, 63, 54, 45, 36, 27, 18, 9, 0,
//  };
//  uint32_t result = 0;
//  int len = startingSquare[square] >= 9 ? 7 - startingSquare[square] / 9
//                                       : 7 - startingSquare[square] % 9;
//  for (int i = 0; i < len; i++) {
//    result += occupied.GetBit(
//                  static_cast<Square>(startingSquare[square] + i * SE + SE))
//              << i;
//  }
//  return result;
//}
//
//std::array<bool, 9> blockPatternToRow(uint32_t blockPattern) {
//  uint32_t block = blockPattern << 1;
//  std::array<bool, 9> result;
//  result.fill(0);
//  uint32_t mask = 1 << 7;
//  for (uint32_t i = 1; i < 8; i++) {
//    if (block & mask)
//      result[i] = 1;
//    mask = mask >> 1;
//  }
//  return result;
//}
//
//std::array<Bitboard, ARRAY_SIZE> initRankAttacks() {
//  std::array<Bitboard, ARRAY_SIZE> result;
//  std::array<bool, BOARD_SIZE> mat;
//
//  for (uint32_t i = 0; i < BOARD_SIZE; i++) {
//    uint32_t columnIdx = i % BOARD_DIM;
//    uint32_t rowIdx = i / BOARD_DIM;
//    for (uint32_t blockPattern = 0; blockPattern < 128; blockPattern++) {
//      mat.fill(0);
//
//      std::array<bool, BOARD_DIM> blockRow = blockPatternToRow(blockPattern);
//      int lastLeft = 0;
//      int firstRight = BOARD_DIM - 1;
//      for (int col = 0; col < columnIdx; col++) {
//        if (blockRow[col]) {
//          lastLeft = col;
//        }
//      }
//      for (int col = columnIdx + 1; col < blockRow.size(); col++) {
//        if (blockRow[col]) {
//          firstRight = col;
//          break;
//        }
//      }
//      for (int blockIdx = 0; blockIdx < blockRow.size(); blockIdx++) {
//        if (blockIdx != columnIdx &&
//            ((blockIdx >= lastLeft && blockIdx < columnIdx) ||
//             (blockIdx <= firstRight && blockIdx > columnIdx)))
//          mat[rowIdx * BOARD_DIM + blockIdx] = 1;
//      }
//      result[i * 128 + blockPattern] = mat;
//    }
//  }
//
//  return result;
//}
//
//std::array<Bitboard, ARRAY_SIZE> initFileAttacks() {
//  std::array<Bitboard, ARRAY_SIZE> result;
//  std::array<bool, BOARD_SIZE> mat;
//
//  for (uint32_t i = 0; i < BOARD_SIZE; i++) {
//    uint32_t columnIdx = i % BOARD_DIM;
//    uint32_t rowIdx = i / BOARD_DIM;
//    for (uint32_t blockPattern = 0; blockPattern < 128; blockPattern++) {
//      mat.fill(0);
//
//      std::array<bool, BOARD_DIM> blockRow = blockPatternToRow(blockPattern);
//      int lastLeft = 0;
//      int firstRight = BOARD_DIM - 1;
//      for (int row = 0; row < rowIdx; row++) {
//        if (blockRow[row]) {
//          lastLeft = row;
//        }
//      }
//      for (int row = rowIdx + 1; row < blockRow.size(); row++) {
//        if (blockRow[row]) {
//          firstRight = row;
//          break;
//        }
//      }
//      for (int blockIdx = 0; blockIdx < blockRow.size(); blockIdx++) {
//        if (blockIdx != rowIdx &&
//            ((blockIdx >= lastLeft && blockIdx < rowIdx) ||
//             (blockIdx <= firstRight && blockIdx > rowIdx)))
//          mat[blockIdx * BOARD_DIM + columnIdx] = 1;
//      }
//      result[i * 128 + blockPattern] = mat;
//    }
//  }
//
//  return result;
//}
//
//std::array<Bitboard, ARRAY_SIZE> initDiagRightAttacks() {
//  std::array<Bitboard, ARRAY_SIZE> result;
//  std::array<bool, BOARD_SIZE> mat;
//
//  for (uint32_t i = 0; i < BOARD_SIZE; i++) {
//    uint32_t columnIdx = i % BOARD_DIM;
//    uint32_t rowIdx = i / BOARD_DIM;
//    int diagLength = (columnIdx + rowIdx < BOARD_DIM)
//                         ? columnIdx + rowIdx + 1
//                         : (BOARD_DIM - columnIdx) + (BOARD_DIM - rowIdx) - 1;
//    uint32_t maxPattern = 1 << std::max((diagLength - 2), 0);
//    for (uint32_t blockPattern = 0; blockPattern < maxPattern; blockPattern++) {
//      mat.fill(0);
//      std::array<bool, BOARD_DIM> blockRow =
//          blockPatternToRow(blockPattern << (BOARD_DIM - diagLength));
//      int tmpColIdx = columnIdx;
//      int tmpRowIdx = rowIdx;
//      int idxInDiag = 0;
//      //// TODO naprawiæ chyba row z col zamienione albo coœ
//      while (tmpColIdx > 0 && tmpRowIdx < BOARD_DIM - 1) {
//        tmpRowIdx++;
//        tmpColIdx--;
//        idxInDiag++;
//      }
//      int lastLeft = 0;
//      int firstRight = diagLength - 1;
//      for (int col = lastLeft; col < idxInDiag; col++) {
//        if (blockRow[col]) {
//          lastLeft = col;
//        }
//      }
//      for (int col = idxInDiag + 1; col < diagLength; col++) {
//        if (blockRow[col]) {
//          firstRight = col;
//          break;
//        }
//      }
//
//      for (int blockIdx = 0; blockIdx < blockRow.size(); blockIdx++) {
//        if (blockIdx != idxInDiag &&
//            ((blockIdx >= lastLeft && blockIdx < idxInDiag) ||
//             (blockIdx <= firstRight && blockIdx > idxInDiag)))
//          mat[tmpRowIdx * BOARD_DIM + tmpColIdx] = 1;
//        if (tmpRowIdx == 0 || tmpColIdx == BOARD_DIM - 1) {
//          break;
//        }
//        tmpRowIdx--;
//        tmpColIdx++;
//      }
//      result[i * 128 + blockPattern] = mat;
//    }
//  }
//
//  return result;
//}
//
//std::array<Bitboard, ARRAY_SIZE> initDiagLeftAttacks() {
//  std::array<Bitboard, ARRAY_SIZE> result;
//  std::array<bool, BOARD_SIZE> mat;
//
//  for (uint32_t i = 0; i < BOARD_SIZE; i++) {
//    uint32_t columnIdx = i % BOARD_DIM;
//    uint32_t rowIdx = i / BOARD_DIM;
//    int diagLength = ((BOARD_DIM - columnIdx) + rowIdx < BOARD_DIM)
//                         ? (BOARD_DIM - columnIdx) + rowIdx
//                         : columnIdx + (BOARD_DIM - rowIdx);
//    uint32_t maxPattern = 1 << std::max((diagLength - 2), 0);
//    for (uint32_t blockPattern = 0; blockPattern < maxPattern; blockPattern++) {
//      mat.fill(0);
//      std::array<bool, BOARD_DIM> blockRow =
//          blockPatternToRow(blockPattern << (BOARD_DIM - diagLength));
//      int tmpColIdx = columnIdx;
//      int tmpRowIdx = rowIdx;
//      int idxInDiag = 0;
//      while (tmpRowIdx < BOARD_DIM - 1 && tmpColIdx < BOARD_DIM - 1) {
//        tmpRowIdx++;
//        tmpColIdx++;
//        idxInDiag++;
//      }
//      int lastLeft = 0;
//      int firstRight = diagLength - 1;
//      for (int col = lastLeft; col < idxInDiag; col++) {
//        if (blockRow[col]) {
//          lastLeft = col;
//        }
//      }
//      for (int col = idxInDiag + 1; col < firstRight; col++) {
//        if (blockRow[col]) {
//          firstRight = col;
//          break;
//        }
//      }
//
//      for (int blockIdx = 0; blockIdx < blockRow.size(); blockIdx++) {
//        if (blockIdx != idxInDiag &&
//            ((blockIdx >= lastLeft && blockIdx < idxInDiag) ||
//             (blockIdx <= firstRight && blockIdx > idxInDiag)))
//          mat[tmpRowIdx * BOARD_DIM + tmpColIdx] = 1;
//        if (tmpRowIdx == 0 || tmpColIdx == 0) {
//          break;
//        }
//        tmpRowIdx--;
//        tmpColIdx--;
//      }
//      result[i * 128 + blockPattern] = mat;
//    }
//  }
//
//  return result;
//}
//
//std::array<Bitboard, BOARD_DIM> initRankMask() {
//  std::array<Bitboard, BOARD_DIM> result;
//  std::array<bool, BOARD_SIZE> mat;
//  mat.fill(0);
//
//  for (int i = 0; i < BOARD_DIM; i++) {
//    for (int j = 0; j < BOARD_DIM; j++) {
//      mat[i * BOARD_DIM + j] = 1;
//    }
//    result[i] = mat;
//  }
//  return result;
//}
//
//std::array<Bitboard, BOARD_DIM> initFileMask() {
//  std::array<Bitboard, BOARD_DIM> result;
//  std::array<bool, BOARD_SIZE> mat;
//  mat.fill(0);
//
//  for (int i = 0; i < BOARD_DIM; i++) {
//    for (int j = 0; j < BOARD_DIM; j++) {
//      mat[j * BOARD_DIM + i] = 1;
//    }
//    result[i] = mat;
//  }
//  return result;
//}
//
//const Bitboard& getRankAttacks(const Square& square, const Bitboard& occupied) {
//  return rankAttacks[square * 128 + getRankBlockPattern(occupied, square)];
//}
//
//const Bitboard& getFileAttacks(const Square& square, const Bitboard& occupied) {
//  return fileAttacks[square * 128 + getFileBlockPattern(occupied, square)];
//}
//const Bitboard& getDiagRightAttacks(const Square& square,
//                                    const Bitboard& occupied) {
//  return diagRightAttacks[square * 128 +
//                          getDiagRightBlockPattern(occupied, square)];
//}
//const Bitboard& getDiagLeftAttacks(const Square& square,
//                                   const Bitboard& occupied) {
//  return diagLeftAttacks[square * 128 +
//                         getDiagLeftBlockPattern(occupied, square)];
//}
//const Bitboard& getRankMask(const uint32_t& rank) {
//  return rankMask[rank];
//}
//const Bitboard& getFileMask(const uint32_t& file) {
//  return fileMask[file];
//}
//
//Bitboard* getRankAttacksPtr() {
//  return rankAttacks.data();
//}
//Bitboard* getFileAttacksPtr() {
//  return fileAttacks.data();
//}
//Bitboard* getDiagRightAttacksPtr() {
//  return diagRightAttacks.data();
//}
//Bitboard* getDiagLeftAttacksPtr() {
//  return diagLeftAttacks.data();
//}
//Bitboard* getRankMaskPtr() {
//  return rankMask.data();
//}
//Bitboard* getFileMaskPtr() {
//  return fileMask.data();
//}
//}  // namespace CPU
//}  // namespace engine
//}  // namespace shogi