#pragma once
#include <vector>
#include "moveGen.h"
#include "moveGenHelpers.h"

namespace shogi {
namespace engine {
namespace CPU {
class MoveList {
 public:
  explicit MoveList(const Board& board, bool isWhite) {
    Bitboard validMoves, attackedByEnemy, pinned;
    try {
      isWhite ? getWhitePiecesInfo(board, pinned, validMoves, attackedByEnemy)
              : getBlackPiecesInfo(board, pinned, validMoves, attackedByEnemy);
    } catch (...) {
      std::cout << "err" << std::endl;
    }
    
    uint32_t count =
        isWhite ? countWhiteMoves(board, pinned, validMoves, attackedByEnemy)
                : countBlackMoves(board, pinned, validMoves, attackedByEnemy);
    moves = std::vector<Move>(count);
    uint32_t generatedCount =
        isWhite ? generateWhiteMoves(board, pinned, validMoves, attackedByEnemy,
                                     data())
                : generateBlackMoves(board, pinned, validMoves, attackedByEnemy,
                                     data());
  }
  Move* data() { return moves.data(); }
  uint32_t size() { return moves.size(); }
  const Move* begin() { return moves.data(); }
  const Move* end() { return moves._Unchecked_end(); }

 private:
  std::vector<Move> moves;
};

template <bool Root, bool Verbose = false>
__host__ uint64_t perft(Board& board,
                        uint16_t depth,
                        std::vector<Move>& movesFromRoot,
                        bool isWhite = false) {
  uint64_t count = 0;
  MoveList moves = MoveList(board, isWhite);
  std::vector<uint64_t> counts;
  if (depth == 1) {
    if constexpr (Root) {
      for (int i = 0; i < moves.size(); i++) {
        if constexpr (Verbose)
          std::cout << moveToUSI(*(moves.data() + i)) << ": " << 1
                    << std::endl;
      }
      if constexpr (Verbose)
        std::cout << "Nodes searched: " << moves.size() << std::endl;
    }
    return moves.size();
  }
  Board oldBoard = board;
  for (const auto& move : moves) {
    movesFromRoot.push_back(move);
    MoveInfo moveReturnInfo = makeMove<true>(board, move);
    if constexpr (Root) {
      counts.push_back(
          perft<false, Verbose>(board, depth - 1, movesFromRoot, !isWhite));
    } else {
      count += perft<false, Verbose>(board, depth - 1, movesFromRoot, !isWhite);
    }
    // unmakeMove(board, move, moveReturnInfo);
    board = oldBoard;
    movesFromRoot.pop_back();
  }
  if constexpr (Root) {
    uint64_t nodesSearched = 0;
    for (int i = 0; i < moves.size(); i++) {
      if constexpr (Verbose)
        std::cout << moveToUSI(*(moves.data() + i)) << ": " << counts[i]
                  << std::endl;
      nodesSearched += counts[i];
    }
    if constexpr (Verbose)
      std::cout << "Nodes searched: " << nodesSearched << std::endl;
    return nodesSearched;
  }
  return count;
}

}  // namespace CPU
}  // namespace engine
}  // namespace shogi