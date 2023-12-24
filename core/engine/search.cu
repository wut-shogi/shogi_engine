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
Move GetBestMove(const Board& board,
                 bool isWhite,
                 uint16_t maxDepth) {
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

Move GetBestMoveAlphaBeta(const Board& board,
                                        bool isWhite,
                                        uint16_t maxDepth) {
  CPU::MoveList rootMoves(board, isWhite);
  std::vector<uint32_t> nodesSearched(maxDepth, 0);
  CPU::MoveList moves(board, isWhite);
  int16_t score = isWhite ? INT16_MIN : INT16_MAX;
  Move bestMove;
  Board newBoard = board;
  for (const auto& move : moves) {
    makeMove(newBoard, move);
    int16_t result = alphaBeta(newBoard, !isWhite, maxDepth-1, INT16_MIN,
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
              << " positions on depth: " << i+1 << std::endl;
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
  Move bestMove;
  bestMove.from = 0;
  bestMove.to = 0;
  bestMove.promotion = 0;
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
        bestMove = move;
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
        bestMove = move;
      }
      if (alpha >= beta) {
        break;
      }
    }
    result = beta;
  }
  return result;
}

}  // namespace SEARCH
}  // namespace engine
}  // namespace shogi