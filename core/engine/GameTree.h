#pragma once
#include <vector>
#include "Board.h"

namespace shogi {
namespace engine {

struct TreeNode;
struct TreeNode2;

Move GetBestMove(uint8_t* d_Buffer, uint32_t d_BufferSize,const Board& board,
                           bool isWhite,
                           uint16_t depth,
                           uint16_t maxDepth);

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

 private:
  Board m_startingBoard;
  bool m_startingIsWhite;
  Bitboard* m_validMoves;
  Bitboard* m_attackedByEnemy;
  uint32_t m_maxProcessedSize = 100000;
  uint32_t m_minBoardsGPU = UINT32_MAX;
  uint32_t m_maxDepth;
  std::vector<uint32_t> positionsSearched;
};
}  // namespace engine
}  // namespace shogi
