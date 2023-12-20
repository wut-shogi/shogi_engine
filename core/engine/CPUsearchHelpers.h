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
    isWhite ? getWhitePiecesInfo(board, pinned, validMoves, attackedByEnemy)
            : getBlackPiecesInfo(board, pinned, validMoves, attackedByEnemy);
    uint32_t count =
        isWhite ? countWhiteMoves(board, pinned, validMoves, attackedByEnemy)
                : countBlackMoves(board, pinned, validMoves, attackedByEnemy);
    moves = std::vector<Move>(count);
    uint32_t generatedCount =
        isWhite ? generateWhiteMoves(board, pinned, validMoves, attackedByEnemy,
                                     data())
                : generateBlackMoves(board, pinned, validMoves, attackedByEnemy,
                                     data());
    if (generatedCount > count) {
      std::cout << "Error" << std::endl;
    }
  }
  Move* data() { return moves.data(); }
  uint32_t size() { return moves.size(); }
  const Move* begin() { return moves.data(); }
  const Move* end() { return moves._Unchecked_end(); }

 private:
  std::vector<Move> moves;
};

template <bool Root>
__host__ uint64_t perft(Board& board, uint16_t depth, bool isWhite = false) {
  uint64_t count = 0;
  MoveList moves = MoveList(board, isWhite);
  std::vector<uint64_t> counts;
  if (depth == 1) {
    return moves.size();
  }
  for (const auto& move : moves) {
    Board oldBoard = board;
    MoveInfo moveReturnInfo = makeMove<true>(board, move);
    if constexpr (Root) {
      counts.push_back(perft<false>(board, depth - 1, !isWhite));
    } else {
      count += perft<false>(board, depth - 1, !isWhite);
    }
    if (move.promotion == 1) {
      std::cout << "Err" << std::endl;
    }
    unmakeMove(board, move, moveReturnInfo);
    for (int i = 0; i < BB::Type::SIZE; i++) {
      if (board[static_cast<BB::Type>(i)] !=
          oldBoard[static_cast<BB::Type>(i)]) {
        std::cout << "Error" << std::endl;
      }
    }
    if (board.inHand.value != oldBoard.inHand.value) {
      std::cout << "Error" << std::endl;
    }
  }
  if constexpr (Root) {
    uint64_t nodesSearched = 0;
    for (int i = 0; i < moves.size(); i++) {
      std::cout << moveToUSI(*(moves.data() + i)) << ": " << counts[i]
                << std::endl;
      nodesSearched += counts[i];
    }
    std::cout << "Nodes searched: " << nodesSearched << std::endl;
  }
  return count;
}

}  // namespace CPU
}  // namespace engine
}  // namespace shogi