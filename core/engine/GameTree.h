#pragma once
#include <vector>
#include "Board.h"

namespace shogi {
namespace engine {

struct TreeNode;
struct TreeNode2;

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

  static std::vector<Move> GetAllMovesFrom(const Board& board, bool isWhite);

  int16_t SearchNode(TreeNode& node);

  /*void SearchNode2(const Board& startBoard,
                      TreeNode2& node,
                      std::vector<int16_t> values);*/

 private:
  Board m_startingBoard;
  bool m_startingIsWhite;
  Bitboard* m_validMoves;
  Bitboard* m_attackedByEnemy;
  uint32_t m_maxProcessedSize = 10000000;
  uint32_t m_minBoardsGPU = 10000;
  uint32_t m_maxDepth;
  std::vector<uint32_t> positionsSearched;
};
}  // namespace engine
}  // namespace shogi
