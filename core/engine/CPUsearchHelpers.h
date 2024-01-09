#pragma once
#include <vector>
#include "MoveGen.h"
#include "MoveGenHelpers.h"

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

    if (count != generatedCount)
      printf(
          "MoveList Error: generated different number of moves "
          "then precounted. Expected %d moves, generated %d moves\n",
          count, generatedCount);
  }
  Move* data() { return moves.data(); }
  uint32_t size() { return moves.size(); }
  const Move* begin() { return moves.data(); }
  const Move* end() { return moves.data() + moves.size(); }

 private:
  std::vector<Move> moves;
};

}  // namespace CPU
}  // namespace engine
}  // namespace shogi