#include <array>
#include <mutex>
#include <vector>
#include "lookUpTables.h"

namespace shogi {
namespace engine {
namespace LookUpTables {
#define ARRAY_SIZE 10368
struct LookUpTables {
  Bitboard* rankAttacks;
  Bitboard* fileAttacks;
  Bitboard* diagRightAttacks;
  Bitboard* diagLeftAttacks;

  Bitboard* rankMask;
  Bitboard* fileMask;

  uint32_t* startSqDiagRight;
  uint32_t* startSqDiagLeft;
};
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

void initRankAttacks(Bitboard*& ptr) {
  std::array<Bitboard, ARRAY_SIZE> result;
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
      result[i * 128 + blockPattern] = mat;
    }
  }
  ptr = new Bitboard[result.size()];
  std::memcpy(ptr, result.data(), result.size() * sizeof(Bitboard));
}

void initFileAttacks(Bitboard*& ptr) {
  std::array<Bitboard, ARRAY_SIZE> result;
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
      result[i * 128 + blockPattern] = mat;
    }
  }
  ptr = new Bitboard[result.size()];
  std::memcpy(ptr, result.data(), result.size() * sizeof(Bitboard));
}

void initDiagRightAttacks(Bitboard*& ptr) {
  std::array<Bitboard, ARRAY_SIZE> result;
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
      //// TODO naprawi� chyba row z col zamienione albo co�
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
      result[i * 128 + blockPattern] = mat;
    }
  }
  ptr = new Bitboard[result.size()];
  std::memcpy(ptr, result.data(), result.size() * sizeof(Bitboard));
}

void initDiagLeftAttacks(Bitboard*& ptr) {
  std::array<Bitboard, ARRAY_SIZE> result;
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
      result[i * 128 + blockPattern] = mat;
    }
  }
  ptr = new Bitboard[result.size()];
  std::memcpy(ptr, result.data(), result.size() * sizeof(Bitboard));
}

void initRankMask(Bitboard*& ptr) {
  std::array<Bitboard, BOARD_DIM> result;
  std::array<bool, BOARD_SIZE> mat;
  mat.fill(0);

  for (int i = 0; i < BOARD_DIM; i++) {
    for (int j = 0; j < BOARD_DIM; j++) {
      mat[i * BOARD_DIM + j] = 1;
    }
    result[i] = mat;
  }
  ptr = new Bitboard[result.size()];
  std::memcpy(ptr, result.data(), result.size() * sizeof(Bitboard));
}

void initFileMask(Bitboard*& ptr) {
  std::array<Bitboard, BOARD_DIM> result;
  std::array<bool, BOARD_SIZE> mat;
  mat.fill(0);

  for (int i = 0; i < BOARD_DIM; i++) {
    for (int j = 0; j < BOARD_DIM; j++) {
      mat[j * BOARD_DIM + i] = 1;
    }
    result[i] = mat;
  }
  ptr = new Bitboard[result.size()];
  std::memcpy(ptr, result.data(), result.size() * sizeof(Bitboard));
}

namespace CPU {
static LookUpTables lookUpTables;
void init() {
  initRankAttacks(CPU::lookUpTables.rankAttacks);
  initFileAttacks(CPU::lookUpTables.fileAttacks);
  initDiagRightAttacks(CPU::lookUpTables.diagRightAttacks);
  initDiagLeftAttacks(CPU::lookUpTables.diagLeftAttacks);
  initRankMask(CPU::lookUpTables.rankMask);
  initFileMask(CPU::lookUpTables.fileMask);
}

void cleanup() {
  delete[] CPU::lookUpTables.rankAttacks;
  delete[] CPU::lookUpTables.fileAttacks;
  delete[] CPU::lookUpTables.diagRightAttacks;
  delete[] CPU::lookUpTables.diagLeftAttacks;
  delete[] CPU::lookUpTables.rankMask;
  delete[] CPU::lookUpTables.fileMask;
}
}  // namespace CPU
namespace GPU {
#ifdef __CUDACC__
__device__ __constant__ LookUpTables lookUpTables[1];
inline static std::vector<LookUpTables> lookUpTables_Host;
inline static std::mutex lookupTablesMutex;
// static LookUpTables lookUpsTables_Host;
#endif
int init(int deviceId) {
#ifdef __CUDACC__
  {
    std::lock_guard<std::mutex> lock(lookupTablesMutex);
    lookUpTables_Host.push_back(LookUpTables());
  }
  cudaMalloc((void**)&lookUpTables_Host[deviceId].rankAttacks,
             ARRAY_SIZE * sizeof(Bitboard));
  cudaMalloc((void**)&lookUpTables_Host[deviceId].fileAttacks,
             ARRAY_SIZE * sizeof(Bitboard));
  cudaMalloc((void**)&lookUpTables_Host[deviceId].diagRightAttacks,
             ARRAY_SIZE * sizeof(Bitboard));
  cudaMalloc((void**)&lookUpTables_Host[deviceId].diagLeftAttacks,
             ARRAY_SIZE * sizeof(Bitboard));
  cudaMalloc((void**)&lookUpTables_Host[deviceId].rankMask,
             9 * sizeof(Bitboard));
  cudaMalloc((void**)&lookUpTables_Host[deviceId].fileMask,
             9 * sizeof(Bitboard));
  cudaMalloc((void**)&lookUpTables_Host[deviceId].startSqDiagRight,
             81 * sizeof(uint32_t));
  cudaMalloc((void**)&lookUpTables_Host[deviceId].startSqDiagLeft,
             81 * sizeof(uint32_t));

  cudaMemcpy(lookUpTables_Host[deviceId].rankAttacks,
             CPU::lookUpTables.rankAttacks, ARRAY_SIZE * sizeof(Bitboard),
             cudaMemcpyHostToDevice);
  cudaMemcpy(lookUpTables_Host[deviceId].fileAttacks,
             CPU::lookUpTables.fileAttacks, ARRAY_SIZE * sizeof(Bitboard),
             cudaMemcpyHostToDevice);
  cudaMemcpy(lookUpTables_Host[deviceId].diagRightAttacks,
             CPU::lookUpTables.diagRightAttacks, ARRAY_SIZE * sizeof(Bitboard),
             cudaMemcpyHostToDevice);
  cudaMemcpy(lookUpTables_Host[deviceId].diagLeftAttacks,
             CPU::lookUpTables.diagLeftAttacks, ARRAY_SIZE * sizeof(Bitboard),
             cudaMemcpyHostToDevice);
  cudaMemcpy(lookUpTables_Host[deviceId].rankMask, CPU::lookUpTables.rankMask,
             9 * sizeof(Bitboard), cudaMemcpyHostToDevice);
  cudaMemcpy(lookUpTables_Host[deviceId].fileMask, CPU::lookUpTables.fileMask,
             9 * sizeof(Bitboard), cudaMemcpyHostToDevice);

  const uint32_t startingSquareDiagRightTemplate[BOARD_SIZE] = {
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
  cudaMemcpy(lookUpTables_Host[deviceId].startSqDiagRight,
             startingSquareDiagRightTemplate, 81 * sizeof(uint32_t),
             cudaMemcpyHostToDevice);

  const uint32_t startingSquareDiagLeftTemplate[BOARD_SIZE] = {
      0,  1,  2,  3,  4,  5,  6,  7, 8,  //
      9,  0,  1,  2,  3,  4,  5,  6, 7,  //
      18, 9,  0,  1,  2,  3,  4,  5, 6,  //
      27, 18, 9,  0,  1,  2,  3,  4, 5,  //
      36, 27, 18, 9,  0,  1,  2,  3, 4,  //
      45, 36, 27, 18, 9,  0,  1,  2, 3,  //
      54, 45, 36, 27, 18, 9,  0,  1, 2,  //
      63, 54, 45, 36, 27, 18, 9,  0, 1,  //
      72, 63, 54, 45, 36, 27, 18, 9, 0,
  };
  cudaMemcpy(lookUpTables_Host[deviceId].startSqDiagLeft,
             startingSquareDiagLeftTemplate, 81 * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  cudaError_t cudaStatus;
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cuda alloc and memcpy launch failed: %s\n",
            cudaGetErrorString(cudaStatus));
    return -1;
  }
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr,
            "cudaDeviceSynchronize returned error code %d after launching "
            "cuda alloc and memcpy!\n",
            cudaStatus);
    return -1;
  }

  cudaMemcpyToSymbol(GPU::lookUpTables, &(lookUpTables_Host[deviceId]),
                     sizeof(LookUpTables), 0, cudaMemcpyHostToDevice);

  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cuda memcpyToSymbol launch failed: %s\n",
            cudaGetErrorString(cudaStatus));
    return -1;
  }
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr,
            "cudaDeviceSynchronize returned error code %d after launching "
            "memcpyToSymbol!\n",
            cudaStatus);
    return -1;
  }
#endif
  return 0;
}

void cleanup(int deviceId) {
#ifdef __CUDACC__
  cudaFree(GPU::lookUpTables_Host[deviceId].rankAttacks);
  cudaFree(GPU::lookUpTables_Host[deviceId].fileAttacks);
  cudaFree(GPU::lookUpTables_Host[deviceId].diagRightAttacks);
  cudaFree(GPU::lookUpTables_Host[deviceId].diagLeftAttacks);
  cudaFree(GPU::lookUpTables_Host[deviceId].rankMask);
  cudaFree(GPU::lookUpTables_Host[deviceId].fileMask);
  cudaFree(GPU::lookUpTables_Host[deviceId].startSqDiagRight);
  cudaFree(GPU::lookUpTables_Host[deviceId].startSqDiagLeft);
#endif
}
}  // namespace GPU

RUNTYPE uint32_t getRankBlockPattern(const Bitboard& bb, Square square) {
  const uint32_t& region = bb[squareToRegion(square)];
  uint32_t rowsBeforeInRegion = (square / BOARD_DIM) % 3;
  uint32_t result = region << 5 << (rowsBeforeInRegion * BOARD_DIM) << 1 >> 25;
  return result;
}

RUNTYPE uint32_t getFileBlockPattern(const Bitboard& occupied, Square square) {
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

RUNTYPE uint32_t getDiagRightBlockPattern(const Bitboard& occupied,
                                          Square square) {
  uint32_t result = 0;
#ifdef __CUDA_ARCH__
  uint32_t startingSquare = GPU::lookUpTables[0].startSqDiagRight[square];
#else
  static const uint32_t startSqDiagRight[BOARD_SIZE] = {
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
  uint32_t startingSquare = startSqDiagRight[square];
#endif
  int len = startingSquare >= 9 ? 7 - startingSquare / 9 : startingSquare - 1;
  for (int i = 0; i < len; i++) {
    result += occupied.GetBit(static_cast<Square>(startingSquare + i * SW + SW))
              << i;
  }
  return result;
}

RUNTYPE uint32_t getDiagLeftBlockPattern(const Bitboard& occupied,
                                         Square square) {
  uint32_t result = 0;
#ifdef __CUDA_ARCH__
  uint32_t startingSquare = GPU::lookUpTables[0].startSqDiagLeft[square];
#else
  static const uint32_t startSqDiagLeft[BOARD_SIZE] = {
      0,  1,  2,  3,  4,  5,  6,  7, 8,  //
      9,  0,  1,  2,  3,  4,  5,  6, 7,  //
      18, 9,  0,  1,  2,  3,  4,  5, 6,  //
      27, 18, 9,  0,  1,  2,  3,  4, 5,  //
      36, 27, 18, 9,  0,  1,  2,  3, 4,  //
      45, 36, 27, 18, 9,  0,  1,  2, 3,  //
      54, 45, 36, 27, 18, 9,  0,  1, 2,  //
      63, 54, 45, 36, 27, 18, 9,  0, 1,  //
      72, 63, 54, 45, 36, 27, 18, 9, 0,
  };
  uint32_t startingSquare = startSqDiagLeft[square];
#endif
  int len =
      startingSquare >= 9 ? 7 - startingSquare / 9 : 7 - startingSquare % 9;
  for (int i = 0; i < len; i++) {
    result += occupied.GetBit(static_cast<Square>(startingSquare + i * SE + SE))
              << i;
  }
  return result;
}

RUNTYPE const Bitboard& getRankAttacks(const Square& square,
                                       const Bitboard& occupied) {
  int index = square * 128 + getRankBlockPattern(occupied, square);
#ifdef __CUDA_ARCH__
  return GPU::lookUpTables[0].rankAttacks[index];
#else
  return CPU::lookUpTables.rankAttacks[index];
#endif
}
RUNTYPE const Bitboard& getFileAttacks(const Square& square,
                                       const Bitboard& occupied) {
  int index = square * 128 + getFileBlockPattern(occupied, square);
#ifdef __CUDA_ARCH__
  return GPU::lookUpTables[0].fileAttacks[index];
#else
  return CPU::lookUpTables.fileAttacks[index];
#endif
}
RUNTYPE const Bitboard& getDiagRightAttacks(const Square& square,
                                            const Bitboard& occupied) {
  int index = square * 128 + getDiagRightBlockPattern(occupied, square);
#ifdef __CUDA_ARCH__
  return GPU::lookUpTables[0].diagRightAttacks[index];
#else
  return CPU::lookUpTables.diagRightAttacks[index];
#endif
}
RUNTYPE const Bitboard& getDiagLeftAttacks(const Square& square,
                                           const Bitboard& occupied) {
  int index = square * 128 + getDiagLeftBlockPattern(occupied, square);
#ifdef __CUDA_ARCH__
  return GPU::lookUpTables[0].diagLeftAttacks[index];
#else
  return CPU::lookUpTables.diagLeftAttacks[index];
#endif
}
RUNTYPE const Bitboard& getRankMask(const uint32_t& rank) {
#ifdef __CUDA_ARCH__
  return GPU::lookUpTables[0].rankMask[rank];
#else
  return CPU::lookUpTables.rankMask[rank];
#endif
}
RUNTYPE const Bitboard& getFileMask(const uint32_t& file) {
#ifdef __CUDA_ARCH__
  return GPU::lookUpTables[0].fileMask[file];
#else
  return CPU::lookUpTables.fileMask[file];
#endif
}
}  // namespace LookUpTables
}  // namespace engine
}  // namespace shogi