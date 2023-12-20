#include "gameTree.h"
#include "GPUsearchHelpers.h"

namespace shogi {
namespace engine {


// True if reached max depth false if not
Move GetBestMove(uint8_t* d_Buffer,
                 uint32_t d_BufferSize,
                 const Board& board,
                 bool isWhite,
                 uint16_t depth,
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
  uint32_t occupiedMemmory = layerSize.back() * depth * sizeof(Move);
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
  std::cout << "Done" << std::endl;
  return bestMove;
}

}  // namespace engine
}  // namespace shogi