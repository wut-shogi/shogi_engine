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

void setDeviceCount(int numberOfDevicesUsed);

Move GetBestMove(const Board& board,
                 bool isWhite,
                 uint16_t maxDepth,
                 uint32_t maxTime,
                 SearchType searchType);

uint64_t countMovesCPU(Board& board, uint16_t depth, bool isWhite);

template <bool Verbose = false>
uint64_t perftCPU(const Board& board, uint16_t depth, bool isWhite = false) {
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

#ifdef __CUDACC__
class GPUBuffer {
 public:
  GPUBuffer(const Board& startBoard, uint8_t* buffer, uint32_t size);

  Board* GetStartBoardPtr();
  bool ReserveMovesSpace(uint32_t size,
                         int16_t movesPerBoard,
                         Move*& outMovesPtr);
  void FreeMovesSpace(Move* moves);
  bool ReserveOffsetsSpace(uint32_t size, uint32_t*& outOffsetsPtr);
  void FreeOffsetsSpace(uint32_t* offsets);
  bool ReserveBitboardsSpace(uint32_t size, uint32_t*& outBitboardsPtr);

  void FreeBitboardsSpace();

 private:
  uint8_t* buffer;
  uint32_t bufferSize;
  Board* d_startBoard;
  uint8_t* freeBegin;
  uint8_t* freeEnd;
};

uint64_t countMovesGPU(bool Verbose,
                       const Board& board,
                       CPU::MoveList& moves,
                       bool isWhite,
                       uint16_t maxDepth);

template <bool Verbose = false>
uint64_t perftGPU(Board& board, uint16_t depth, bool isWhite = false) {
  GPUBuffer gpuBuffer(board, nullptr, 0);
  CPU::MoveList moves(board, isWhite);
  uint64_t nodesSearched = countMovesGPU(Verbose, board, moves, isWhite, depth);
  if constexpr (Verbose)
    std::cout << "Nodes searched: " << nodesSearched << std::endl;
  return nodesSearched;
}
#endif
template <bool Verbose = false>
uint64_t perft(Board& board,
               bool isWhite,
               uint16_t depth,
               SearchType searchType) {
  if (searchType == GPU) {
#ifdef __CUDACC__
    return perftGPU<Verbose>(board, depth, isWhite);
#else
    return 0;
#endif  // __CUDACC__
  } else {
    return perftCPU<Verbose>(board, depth, isWhite);
  }
}
}  // namespace SEARCH
}  // namespace engine
}  // namespace shogi
