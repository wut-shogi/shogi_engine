#pragma once
#include <vector>
#include "Board.h"

namespace shogi {
namespace engine {

struct TreeNode;

class GameTree {
 public:
  GameTree(const Board& board, bool isWhite, int maxDepth)
      : m_startingBoard(board),
        m_startingIsWhite(isWhite),
        m_maxDepth(maxDepth) {
    m_validMoves = new Bitboard[m_maxProcessedSize];
    m_attackedByEnemy = new Bitboard[m_maxProcessedSize];
  }
  Move FindBestMove();

  int16_t SearchNode(TreeNode& node);

 private:
  Board m_startingBoard;
  bool m_startingIsWhite;
  /* Board* m_currentBoards;
   uint32_t boardsLength;*/
  Bitboard* m_validMoves;
  Bitboard* m_attackedByEnemy;
  // Board* m_nextBoards;
  uint32_t m_maxProcessedSize = 10000000;
  uint32_t m_minBoardsGPU = UINT32_MAX;
  uint32_t m_maxDepth;
  std::vector<uint32_t> positionsSearched;
};
}  // namespace engine
}  // namespace shogi
