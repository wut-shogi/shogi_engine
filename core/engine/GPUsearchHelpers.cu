#include "GPUsearchHelpers.h"
#include <device_launch_parameters.h>
#include <stdio.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include "moveGen.h"
#include "moveGenHelpers.h"
#include "evaluation.h"

namespace shogi {
namespace engine {

#define THREADS_COUNT 32

namespace GPU {
#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code,
                      const char* file,
                      int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}
}  // namespace GPU

int calculateNumberOfBlocks(uint32_t size) {
  int numberOfBlocks = (int)ceil(size / (double)THREADS_COUNT);
  return numberOfBlocks;
}

__global__ void countWhiteMovesKernel(uint32_t size,
                                      int16_t movesPerBoard,
                                      Board* startBoard,
                                      Move* inMoves,
                                      uint32_t* outOffsets,
                                      uint32_t* outBitboards) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size)
    return;

  Board board = *startBoard;
  for (int m = 0; m < movesPerBoard; m++) {
    makeMove(board, inMoves[m * size + index]);
  }

  Bitboard pinned, validMoves, attackedByEnemy;
  getWhitePiecesInfo(board, pinned, validMoves, attackedByEnemy);
  uint32_t numberOfMoves =
      countWhiteMoves(board, pinned, validMoves, attackedByEnemy);

  outBitboards[index] = validMoves[TOP];
  outBitboards[size + index] = validMoves[MID];
  outBitboards[2 * size + index] = validMoves[BOTTOM];
  outBitboards[3 * size + index] = attackedByEnemy[TOP];
  outBitboards[4 * size + index] = attackedByEnemy[MID];
  outBitboards[5 * size + index] = attackedByEnemy[BOTTOM];
  outBitboards[6 * size + index] = pinned[TOP];
  outBitboards[7 * size + index] = pinned[MID];
  outBitboards[8 * size + index] = pinned[BOTTOM];
  outOffsets[index] = numberOfMoves;
}

__global__ void countBlackMovesKernel(uint32_t size,
                                      int16_t movesPerBoard,
                                      Board* startBoard,
                                      Move* inMoves,
                                      uint32_t* outOffsets,
                                      uint32_t* outBitboards) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size)
    return;

  Board board = *startBoard;
  for (int m = 0; m < movesPerBoard; m++) {
    makeMove(board, inMoves[m * size + index]);
  }

  Bitboard pinned, validMoves, attackedByEnemy;
  getBlackPiecesInfo(board, pinned, validMoves, attackedByEnemy);
  uint32_t numberOfMoves =
      countBlackMoves(board, pinned, validMoves, attackedByEnemy);

  outBitboards[index] = validMoves[TOP];
  outBitboards[size + index] = validMoves[MID];
  outBitboards[2 * size + index] = validMoves[BOTTOM];
  outBitboards[3 * size + index] = attackedByEnemy[TOP];
  outBitboards[4 * size + index] = attackedByEnemy[MID];
  outBitboards[5 * size + index] = attackedByEnemy[BOTTOM];
  outBitboards[6 * size + index] = pinned[TOP];
  outBitboards[7 * size + index] = pinned[MID];
  outBitboards[8 * size + index] = pinned[BOTTOM];
  outOffsets[index] = numberOfMoves;
}

int GPU::prefixSum(uint32_t* inValues, uint32_t inValuesLength) {
  thrust::exclusive_scan(thrust::device, inValues, inValues + inValuesLength,
                         inValues);
  cudaError_t cudaStatus;
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "countBlackMoves launch failed: %s\n",
            cudaGetErrorString(cudaStatus));
    return -1;
  }
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr,
            "cudaDeviceSynchronize returned error code %d after launching "
            "countBlackMoves!\n",
            cudaStatus);
    return -1;
  }
  return 0;
}

__global__ void generateWhiteMovesKernel(uint32_t size,
                                         int16_t movesPerBoard,
                                         Board* startBoard,
                                         Move* inMoves,
                                         uint32_t* inOffsets,
                                         uint32_t* inBitboards,
                                         Move* outMoves) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size)
    return;

  Board board = *startBoard;
  uint32_t movesOffset = inOffsets[index];
  uint32_t numberOfMoves = inOffsets[index + 1] - movesOffset;
  uint32_t allMovesSize = inOffsets[size] - inOffsets[0];
  for (int m = 0; m < movesPerBoard; m++) {
    Move move = inMoves[m * size + index];
    makeMove(board, move);
    for (int i = 0; i < numberOfMoves; i++) {
      outMoves[movesOffset + i] = move;
    }
    movesOffset += allMovesSize;
  }
  Bitboard validMoves, pinned, attackedByEnemy;
  validMoves[TOP] = inBitboards[index];
  validMoves[MID] = inBitboards[size + index];
  validMoves[BOTTOM] = inBitboards[2 * size + index];
  attackedByEnemy[TOP] = inBitboards[3 * size + index];
  attackedByEnemy[MID] = inBitboards[4 * size + index];
  attackedByEnemy[BOTTOM] = inBitboards[5 * size + index];
  pinned[TOP] = inBitboards[6 * size + index];
  pinned[MID] = inBitboards[7 * size + index];
  pinned[BOTTOM] = inBitboards[8 * size + index];
  

  uint32_t numberOfGeneratedMoves = generateWhiteMoves(board, pinned, validMoves, attackedByEnemy,
                     outMoves + movesOffset);

  if (numberOfGeneratedMoves != numberOfMoves) {
    printf(
        "generateWhiteMovesKernel Error: generated different number of moves then precounted");
  }
}

__global__ void generateBlackMovesKernel(uint32_t size,
                                         int16_t movesPerBoard,
                                         Board* startBoard,
                                         Move* inMoves,
                                         uint32_t* inOffsets,
                                         uint32_t* inBitboards,
                                         Move* outMoves) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size)
    return;

  Board board = *startBoard;
  uint32_t movesOffset = inOffsets[index];
  uint32_t numberOfMoves = inOffsets[index + 1] - movesOffset;
  uint32_t allMovesSize = inOffsets[size] - inOffsets[0];
  for (int m = 0; m < movesPerBoard; m++) {
    Move move = inMoves[m * size + index];
    makeMove(board, move);
    for (int i = 0; i < numberOfMoves; i++) {
      outMoves[movesOffset + i] = move;
    }
    movesOffset += allMovesSize;
  }
  Bitboard validMoves, pinned, attackedByEnemy;
  validMoves[TOP] = inBitboards[index];
  validMoves[MID] = inBitboards[size + index];
  validMoves[BOTTOM] = inBitboards[2 * size + index];
  attackedByEnemy[TOP] = inBitboards[3 * size + index];
  attackedByEnemy[MID] = inBitboards[4 * size + index];
  attackedByEnemy[BOTTOM] = inBitboards[5 * size + index];
  pinned[TOP] = inBitboards[6 * size + index];
  pinned[MID] = inBitboards[7 * size + index];
  pinned[BOTTOM] = inBitboards[8 * size + index];

  uint32_t numberOfGeneratedMoves = generateBlackMoves(
      board, pinned, validMoves, attackedByEnemy, outMoves + movesOffset);


  if (numberOfGeneratedMoves != numberOfMoves) {
    printf(
        "generateBlackMovesKernel Error: generated different number of moves then precounted");
  }
}

__global__ void evaluateBoardsKernel(uint32_t size,
                                     int16_t movesPerBoard,
                                     Board* startBoard,
                                     Move* inMoves,
                                     int16_t* outValues) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size)
    return;

  Board board = *startBoard;
  for (int m = 0; m < movesPerBoard; m++) {
    makeMove(board, inMoves[m * size + index]);
  }

  outValues[index] = evaluate(board);
}

__global__ void gatherValuesMaxKernel(uint32_t size,
                                      uint16_t depth,
                                      uint32_t* inOffsets,
                                      int16_t* inValues,
                                      int16_t* outValues,
                                      uint32_t* bestIndex) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size)
    return;
  uint32_t childrenStart = inOffsets[index];
  uint32_t childrenSize = inOffsets[index + 1] - childrenStart;
  int16_t maxValue = INT16_MIN;
  for (int i = 0; i < childrenSize; i++) {
    int16_t value = inValues[childrenStart + i];
    if (value > maxValue) {
      maxValue = value;
      if (index == 0) {
        *bestIndex = i;
      }
    }
  }
  if (childrenSize == 0) {
    maxValue = -PieceValue::MATE + depth;
  }
  outValues[index] = maxValue;
}

__global__ void gatherValuesMinKernel(uint32_t size,
                                      uint16_t depth,
                                      uint32_t* inOffsets,
                                      int16_t* inValues,
                                      int16_t* outValues,
                                      uint32_t* bestIndex) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size)
    return;
  uint32_t childrenStart = inOffsets[index];
  uint32_t childrenSize = inOffsets[index + 1] - childrenStart;
  int16_t minValue = INT16_MAX;
  for (int i = 0; i < childrenSize; i++) {
    int16_t value = inValues[childrenStart + i];
    if (value < minValue) {
      minValue = value;
      if (index == 0) {
        *bestIndex = i;
      }
    }
  }
  if (childrenSize == 0) {
    minValue = PieceValue::MATE - depth;
  }
  outValues[index] = minValue;
}

int GPU::countWhiteMoves(uint32_t size,
                         int16_t movesPerBoard,
                         Board* startBoard,
                         Move* inMoves,
                         uint32_t* outOffsets,
                         uint32_t* outBitboards) {
  int numberOfBlocks = calculateNumberOfBlocks(size);
  countWhiteMovesKernel<<<numberOfBlocks, THREADS_COUNT>>>(
      size, movesPerBoard, startBoard, inMoves, outOffsets, outBitboards);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return 0;
}

int GPU::countBlackMoves(uint32_t size,
                         int16_t movesPerBoard,
                         Board* startBoard,
                         Move* inMoves,
                         uint32_t* outOffsets,
                         uint32_t* outBitboards) {
  int numberOfBlocks = calculateNumberOfBlocks(size);
  countBlackMovesKernel<<<numberOfBlocks, THREADS_COUNT>>>(
      size, movesPerBoard, startBoard, inMoves, outOffsets, outBitboards);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return 0;
}

int GPU::generateWhiteMoves(uint32_t size,
                            int16_t movesPerBoard,
                            Board* startBoard,
                            Move* inMoves,
                            uint32_t* inOffsets,
                            uint32_t* inBitboards,
                            Move* outMoves) {
  int numberOfBlocks = calculateNumberOfBlocks(size);
  generateWhiteMovesKernel<<<numberOfBlocks, THREADS_COUNT>>>(
      size, movesPerBoard, startBoard, inMoves, inOffsets, inBitboards,
      outMoves);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return 0;
}

int GPU::generateBlackMoves(uint32_t size,
                            int16_t movesPerBoard,
                            Board* startBoard,
                            Move* inMoves,
                            uint32_t* inOffsets,
                            uint32_t* inBitboards,
                            Move* outMoves) {
  int numberOfBlocks = calculateNumberOfBlocks(size);
  generateBlackMovesKernel<<<numberOfBlocks, THREADS_COUNT>>>(
      size, movesPerBoard, startBoard, inMoves, inOffsets, inBitboards,
      outMoves);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return 0;
}

int GPU::evaluateBoards(uint32_t size,
                        int16_t movesPerBoard,
                        Board* startBoard,
                        Move* inMoves,
                        int16_t* outValues) {
  int numberOfBlocks = calculateNumberOfBlocks(size);
  evaluateBoardsKernel<<<numberOfBlocks, THREADS_COUNT>>>(
      size, movesPerBoard, startBoard, inMoves, outValues);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return 0;
}

int GPU::gatherValuesMax(uint32_t size,
                         uint16_t depth,
                         uint32_t* inOffsets,
                         int16_t* inValues,
                         int16_t* outValues,
                         uint32_t* bestIndex) {
  int numberOfBlocks = calculateNumberOfBlocks(size);
  gatherValuesMaxKernel<<<numberOfBlocks, THREADS_COUNT>>>(
      size, depth, inOffsets, inValues, outValues, bestIndex);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return 0;
}

int GPU::gatherValuesMin(uint32_t size,
                         uint16_t depth,
                         uint32_t* inOffsets,
                         int16_t* inValues,
                         int16_t* outValues,
                         uint32_t* bestIndex) {
  int numberOfBlocks = calculateNumberOfBlocks(size);
  gatherValuesMinKernel<<<numberOfBlocks, THREADS_COUNT>>>(
      size, depth, inOffsets, inValues, outValues, bestIndex);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return 0;
}
}  // namespace engine
}  // namespace shogi