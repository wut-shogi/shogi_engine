#pragma once
#include <vector>
#include "Board.h"
#include "CPUsearchHelpers.h"
#include "USIconverter.h"

namespace shogi {
namespace engine {
namespace SEARCH {

enum SearchType { CPU, GPU };

bool init();

void cleanup();

Move GetBestMove(const Board& board,
                 bool isWhite,
                 uint16_t maxDepth,
                 uint32_t maxTime,
                 SearchType searchType);

uint64_t countMovesCPU(Board& board, uint16_t depth, bool isWhite);

template <bool Verbose = false>
__host__ uint64_t perftCPU(const Board& board,
                           uint16_t depth,
                           bool isWhite = false) {
  CPU::MoveList moves = CPU::MoveList(board, isWhite);
  if (depth == 1) {
    for (int i = 0; i < moves.size(); i++) {
      if constexpr (Verbose)
        std::cout << MoveToUSI(*(moves.data() + i)) << ": " << 1 << std::endl;
    }
    if constexpr (Verbose)
      std::cout << "Nodes searched: " << moves.size() << std::endl;
    return moves.size();
  }
  uint64_t moveCount = 0;
  uint64_t count;
  Board tmpBoard = board;
  for (const auto& move : moves) {
    MoveInfo moveReturnInfo = makeMove<true>(tmpBoard, move);
    count = countMovesCPU(tmpBoard, depth - 1, !isWhite);
    tmpBoard = board;
    if constexpr (Verbose)
      std::cout << MoveToUSI(move) << ": " << count << std::endl;
    moveCount += count;
  }
  if constexpr (Verbose)
    std::cout << "Nodes searched: " << moveCount << std::endl;
  return moveCount;
}

class GPUBuffer {
 public:
  GPUBuffer(const Board& startBoard);

  Board* GetStartBoardPtr();
  bool ReserveMovesSpace(uint32_t size,
                         int16_t movesPerBoard,
                         Move*& outMovesPtr);
  void FreeMovesSpace(Move* moves);
  bool ReserveOffsetsSpace(uint32_t size, uint32_t*& outOffsetsPtr);
  void FreeOffsetsSpace(uint32_t* offsets);
  bool ReserveBitboardsSpace(uint32_t size, uint32_t*& outBitboardsPtr);

  void FreeBitboardsSpace(uint32_t size);

 private:
  Board* d_startBoard;
  uint8_t* freeBegin;
  uint8_t* freeEnd;
};

uint64_t countMovesGPU(Move* moves,
                       uint32_t size,
                       bool isWhite,
                       uint16_t depth,
                       uint16_t maxDepth,
                       GPUBuffer& gpuBuffer);

template <bool Verbose = false>
__host__ uint64_t perftGPU(Board& board, uint16_t depth, bool isWhite = false) {
  GPUBuffer gpuBuffer(board);
  CPU::MoveList moves(board, isWhite);
  uint64_t nodesSearched = 0;
  for (int i = 0; i < moves.size(); i++) {
    Move* d_moves;
    gpuBuffer.ReserveMovesSpace(1, 1, d_moves);
    cudaMemcpy(d_moves, moves.begin() + i, sizeof(Move),
               cudaMemcpyHostToDevice);
    uint64_t numberOfMoves =
        countMovesGPU(d_moves, 1, !isWhite, 1, depth, gpuBuffer);
    if constexpr (Verbose)
      std::cout << MoveToUSI(*(moves.data() + i)) << ": " << numberOfMoves
                << std::endl;
    nodesSearched += numberOfMoves;
  }
  if constexpr (Verbose)
    std::cout << "Nodes searched: " << nodesSearched << std::endl;
  return nodesSearched;
}

}  // namespace SEARCH
}  // namespace engine
}  // namespace shogi
