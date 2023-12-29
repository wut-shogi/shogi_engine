#include <thrust/extrema.h>
#include "CPUsearchHelpers.h"
#include "GPUsearchHelpers.h"
#include "evaluation.h"
#include "lookUpTables.h"
#include "search.h"

namespace shogi {
namespace engine {
namespace SEARCH {

uint8_t* d_Buffer;
uint32_t d_BufferSize = 0;

bool init() {
  try {
    LookUpTables::CPU::init();
    LookUpTables::GPU::init();
    size_t total = 0, free = 0;
    cudaMemGetInfo(&free, &total);
    if (free == 0)
      return false;
    d_BufferSize = (free / 4) * 4;
    cudaError_t error = cudaMalloc((void**)&d_Buffer, d_BufferSize);
    if (error != cudaSuccess)
      return false;
  } catch (...) {
    return false;
  }
  return true;
}

void cleanup() {
  LookUpTables::CPU::cleanup();
  LookUpTables::GPU::cleanup();
  if (d_BufferSize > 0)
    cudaFree(d_Buffer);
}

// True if reached max depth false if not
Move GetBestMove(const Board& board, bool isWhite, uint16_t maxDepth) {
  Board* d_Board = (Board*)d_Buffer;
  cudaMemcpy(d_Board, &board, sizeof(Board), cudaMemcpyHostToDevice);
  uint8_t* bufferBegin = d_Buffer + sizeof(Board);
  uint8_t* bufferEnd = d_Buffer + d_BufferSize;
  Move* movesPtr = (Move*)(bufferEnd);
  uint32_t* offsetsPtr = (uint32_t*)bufferBegin;
  std::vector<uint32_t> layerSize;
  layerSize.push_back(1);
  // To count how much moves it will generate we need
  // (size+1) * sizeof(uint32_t) + 3 * 3 * size * sizeof(uint32_t) +
  // allMovesSize * depth * sizeof(Move)
  uint32_t occupiedMemmory = 0;
  uint16_t depth = 0;
  while (depth < maxDepth) {
    occupiedMemmory += (layerSize.back() + 1) * sizeof(uint32_t);
    if (occupiedMemmory + 3 * 3 * layerSize.back() * sizeof(uint32_t) >=
        d_BufferSize) {
      break;
    }
    uint32_t* tmpBitboardsPtr = offsetsPtr + layerSize.back() + 1;
    // Count next moves
    isWhite ? GPU::countWhiteMoves(layerSize.back(), depth, d_Board, movesPtr,
                                   offsetsPtr, tmpBitboardsPtr)
            : GPU::countBlackMoves(layerSize.back(), depth, d_Board, movesPtr,
                                   offsetsPtr, tmpBitboardsPtr);
    GPU::prefixSum(offsetsPtr, layerSize.back() + 1);
    uint32_t nextLayerSize = 0;
    cudaMemcpy(&nextLayerSize, offsetsPtr + layerSize.back(), sizeof(uint32_t),
               cudaMemcpyDeviceToHost);

    occupiedMemmory += nextLayerSize * (depth + 1) * sizeof(Move);
    if (occupiedMemmory + 3 * 3 * layerSize.back() * sizeof(uint32_t) >=
        d_BufferSize) {
      break;
    }

    Move* newMovesPtr = movesPtr - nextLayerSize * (depth + 1);
    isWhite
        ? GPU::generateWhiteMoves(layerSize.back(), depth, d_Board, movesPtr,
                                  offsetsPtr, tmpBitboardsPtr, newMovesPtr)
        : GPU::generateBlackMoves(layerSize.back(), depth, d_Board, movesPtr,
                                  offsetsPtr, tmpBitboardsPtr, newMovesPtr);

    offsetsPtr += layerSize.back() + 1;
    movesPtr = newMovesPtr;
    layerSize.push_back(nextLayerSize);
    isWhite = !isWhite;
    depth++;
    std::cout << "Generated: " << layerSize.back()
              << " positions on depth: " << depth << std::endl;
  }
  // After filling up space evaluate boards
  // We can place values in place of last values
  int16_t* valuesPtr = (int16_t*)movesPtr;
  uint32_t valuesOffset = layerSize.back();
  GPU::evaluateBoards(layerSize.back(), depth, d_Board, movesPtr, valuesPtr);
  // We can use board memmory for index
  uint32_t* bestIndex = (uint32_t*)d_Board;
  // Collect values from upper layers
  for (uint16_t d = depth; d > 0; d--) {
    isWhite = !isWhite;
    offsetsPtr -= layerSize[d - 1] + 1;
    isWhite ? GPU::gatherValuesMax(layerSize[d - 1], d, offsetsPtr, valuesPtr,
                                   valuesPtr + valuesOffset, bestIndex)
            : GPU::gatherValuesMin(layerSize[d - 1], d, offsetsPtr, valuesPtr,
                                   valuesPtr + valuesOffset, bestIndex);
    valuesPtr += valuesOffset;
  }
  int16_t bestValue;
  cudaMemcpy(&bestValue, valuesPtr, sizeof(int16_t), cudaMemcpyDeviceToHost);
  uint32_t h_bestIndex;
  cudaMemcpy(&h_bestIndex, bestIndex, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  Move bestMove;
  movesPtr = (Move*)(bufferEnd - layerSize[1] * sizeof(Move));
  cudaMemcpy(&bestMove, movesPtr + h_bestIndex, sizeof(Move),
             cudaMemcpyDeviceToHost);
  return bestMove;
}

Move GetBestMoveAlphaBeta(const Board& board, bool isWhite, uint16_t maxDepth) {
  CPU::MoveList rootMoves(board, isWhite);
  std::vector<uint32_t> nodesSearched(maxDepth, 0);
  CPU::MoveList moves(board, isWhite);
  int16_t score = isWhite ? INT16_MIN : INT16_MAX;
  Move bestMove;
  Board newBoard = board;
  for (const auto& move : moves) {
    makeMove(newBoard, move);
    int16_t result = alphaBeta(newBoard, !isWhite, maxDepth - 1, INT16_MIN,
                               INT16_MAX, nodesSearched);
    newBoard = board;
    if ((isWhite && result > score) || (!isWhite && result < score)) {
      bestMove = move;
      score = result;
    }
  }
  std::cout << "Generated: " << moves.size() << " positions on depth: " << 1
            << std::endl;
  for (int i = 1; i < nodesSearched.size(); i++) {
    std::cout << "Generated: " << nodesSearched[i]
              << " positions on depth: " << i + 1 << std::endl;
  }
  return bestMove;
}

int16_t alphaBeta(Board& board,
                  bool isWhite,
                  uint16_t depth,
                  int16_t alpha,
                  int16_t beta,
                  std::vector<uint32_t>& nodesSearched) {
  CPU::MoveList moves(board, isWhite);
  if (moves.size() == 0) {
    return isWhite ? INT16_MIN : INT16_MAX;
  }
  if (depth == 0) {
    return evaluate(board);
  }
  Board oldBoard = board;
  int16_t result = 0;
  if (isWhite) {
    result = INT16_MIN;
    for (const Move& move : moves) {
      makeMove(board, move);
      result =
          alphaBeta(board, !isWhite, depth - 1, alpha, beta, nodesSearched);
      board = oldBoard;
      nodesSearched[nodesSearched.size() - depth]++;
      if (result > alpha) {
        alpha = result;
      }
      if (alpha >= beta) {
        break;
      }
    }
    result = alpha;
  } else {
    for (const Move& move : moves) {
      makeMove(board, move);
      result =
          alphaBeta(board, !isWhite, depth - 1, alpha, beta, nodesSearched);
      board = oldBoard;
      nodesSearched[nodesSearched.size() - depth]++;
      if (result < beta) {
        beta = result;
      }
      if (alpha >= beta) {
        break;
      }
    }
    result = beta;
  }
  return result;
}

static const uint32_t maxProcessedSize = 5000;

class GPUBuffer {
 public:
  GPUBuffer(const Board& startBoard) {
    d_startBoard = (Board*)d_Buffer;
    cudaMemcpy(d_startBoard, &startBoard, sizeof(Board),
               cudaMemcpyHostToDevice);
    freeBegin = d_Buffer + sizeof(Board);
    freeEnd = d_Buffer + d_BufferSize;
  }

  Board* GetStartBoardPtr() { return d_startBoard; }
  bool ReserveMovesSpace(uint32_t size,
                         int16_t movesPerBoard,
                         Move*& outMovesPtr) {
    outMovesPtr = (Move*)freeBegin;
    freeBegin += size * movesPerBoard * sizeof(Move);
    return freeBegin < freeEnd;
  }
  void FreeMovesSpace(Move* moves) { freeBegin = (uint8_t*)moves; }
  bool ReserveOffsetsSpace(uint32_t size, uint32_t*& outOffsetsPtr) {
    uint32_t aligmentMismatch = (size_t)freeBegin % 4;
    if (aligmentMismatch != 0) {
      freeBegin += 4 - aligmentMismatch;
    }
    outOffsetsPtr = (uint32_t*)freeBegin;
    freeBegin += size * sizeof(uint32_t);
    return freeBegin < freeEnd;
  }
  void FreeOffsetsSpace(uint32_t* offsets) { freeBegin = (uint8_t*)offsets; }
  bool ReserveBitboardsSpace(uint32_t size, uint32_t*& outBitboardsPtr) {
    // It is temporary so we allocate it from the back so it can be freed easily
    freeEnd -= size * 3 * 3 * sizeof(uint32_t);
    outBitboardsPtr = (uint32_t*)freeEnd;
    return freeBegin < freeEnd;
  }

  void FreeBitboardsSpace(uint32_t size) { freeEnd = d_Buffer + d_BufferSize; }

 private:
  Board* d_startBoard;
  uint8_t* freeBegin;
  uint8_t* freeEnd;
};

// Converts moves to values
void minMaxGPU(Move* moves,
               uint32_t size,
               bool isWhite,
               uint16_t depth,
               uint16_t maxDepth,
               GPUBuffer& gpuBuffer,
    std::vector<uint32_t>& numberOfMovesPerDepth) {
  if (depth == maxDepth) {
    // Evaluate moves
    GPU::evaluateBoards(size, depth, gpuBuffer.GetStartBoardPtr(), moves,
                        (int16_t*)moves);
    numberOfMovesPerDepth[depth - 1] += size;
    return;
  }
  uint32_t processed = 0;
  uint32_t* offsets;
  uint32_t* bitboards;
  uint32_t* bestIndex;
  // Process by chunks
  while (processed < size) {
    uint32_t sizeToProcess = std::min(size - processed, maxProcessedSize);
    // Calculate offsets
    if (!gpuBuffer.ReserveOffsetsSpace(sizeToProcess + 1, offsets))
      printf("Err in ReserveOffsetsSpace\n");
    if (!gpuBuffer.ReserveBitboardsSpace(sizeToProcess, bitboards))
      printf("Err in ReserveBitboardsSpace\n");
    isWhite ? GPU::countWhiteMoves(sizeToProcess, depth,
                                   gpuBuffer.GetStartBoardPtr(), moves, size,
                                   processed, offsets, bitboards)
            : GPU::countBlackMoves(sizeToProcess, depth,
                                   gpuBuffer.GetStartBoardPtr(), moves, size,
                                   processed, offsets, bitboards);
    GPU::prefixSum(offsets, sizeToProcess + 1);
    uint32_t nextLayerSize = 0;
    cudaMemcpy(&nextLayerSize, offsets + sizeToProcess, sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    // Generate new moves
    Move* newMoves;

    if (!gpuBuffer.ReserveMovesSpace(nextLayerSize, depth + 1, newMoves))
      printf("Err in ReserveMovesSpace\n");
    isWhite ? GPU::generateWhiteMoves(sizeToProcess, depth,
                                      gpuBuffer.GetStartBoardPtr(), moves, size,
                                      processed, offsets, bitboards, newMoves)
            : GPU::generateBlackMoves(sizeToProcess, depth,
                                      gpuBuffer.GetStartBoardPtr(), moves, size,
                                      processed, offsets, bitboards, newMoves);
    gpuBuffer.FreeBitboardsSpace(sizeToProcess);
    // minmaxGPU(newLayer)
    minMaxGPU(newMoves, nextLayerSize, !isWhite, depth + 1, maxDepth, gpuBuffer,
              numberOfMovesPerDepth);
    bestIndex = offsets;
    // Gather values from new layer
    isWhite ? GPU::gatherValuesMax(sizeToProcess, depth, offsets,
                                   (int16_t*)newMoves,
                                   (int16_t*)(moves + processed), bestIndex)
            : GPU::gatherValuesMin(sizeToProcess, depth, offsets,
                                   (int16_t*)newMoves,
                                   (int16_t*)(moves + processed), bestIndex);
    gpuBuffer.FreeMovesSpace(newMoves);
    gpuBuffer.FreeOffsetsSpace(offsets);
    processed += sizeToProcess;
  }
  numberOfMovesPerDepth[depth - 1] += size;
}

Move GetBestMove2(const Board& board, bool isWhite, uint16_t maxDepth) {
  std::vector<uint32_t> numberOfMovesPerDepth(maxDepth, 0);
  GPUBuffer gpuBuffer(board);
  CPU::MoveList rootMoves(board, isWhite);
  Move* d_moves;
  gpuBuffer.ReserveMovesSpace(rootMoves.size(), 1, d_moves);
  cudaMemcpy(d_moves, rootMoves.begin(), rootMoves.size() * sizeof(Move),
             cudaMemcpyHostToDevice);
  minMaxGPU(d_moves, rootMoves.size(), !isWhite, 1, maxDepth, gpuBuffer, numberOfMovesPerDepth);
  std::vector<int16_t> h_values(rootMoves.size());
  cudaMemcpy(h_values.data(), d_moves, rootMoves.size() * sizeof(int16_t),
             cudaMemcpyDeviceToHost);
  size_t bestValueIdx =
      isWhite ? std::max_element(h_values.begin(), h_values.end()) -
                    h_values.begin()
              : std::min_element(h_values.begin(), h_values.end()) -
                    h_values.begin();

  for (int i = 1; i < numberOfMovesPerDepth.size(); i++) {
    std::cout << "Generated: " << numberOfMovesPerDepth[i]
              << " positions on depth: " << i + 1 << std::endl;
  }
  return *(rootMoves.begin() + bestValueIdx);
}
}  // namespace SEARCH
}  // namespace engine
}  // namespace shogi