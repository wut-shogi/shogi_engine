#pragma once
#include <vector>
#include "MoveGen.h"

namespace shogi {
namespace engine {
struct TreeLevel {
  Board* boardsArray;
  uint32_t length;
  Move* movesArray;
  bool isWhite;
  int depth;
};
void prefixSum(uint32_t* array, int length);

void benchmarkTreeBuilding(int maxDepth);

TreeLevel buildNextLevel(TreeLevel& level);

struct TreeNode {
  std::vector<Board> boards;
  std::vector<Move> moves;
  std::vector<int16_t> values;
  uint32_t movesCount;
  bool isWhite;
  uint32_t depth;
};

class GameTree {
 public:
  GameTree(const Board& board, bool isWhite, int maxDepth)
      : m_startingBoard(board),
        m_startingIsWhite(isWhite),
        m_maxDepth(maxDepth) {
    m_validMoves = new Bitboard[m_maxProcessedSize];
    m_attackedByEnemy = new Bitboard[m_maxProcessedSize];
  }
  void FindBestMove() {
    TreeNode root;
    root.isWhite = m_startingIsWhite;
    root.depth = m_maxDepth;
    root.boards = {m_startingBoard};
    positionsSearched = std::vector<uint32_t>(m_maxDepth, 0);
    int16_t value;
    SearchNode(root, &value);
    for (int i = m_maxDepth - 1; i >= 0; i--) {
      std::cout << "Positions on depth " << i << " : " << positionsSearched[i]
                << std::endl;
    }
  }

  void SearchNode(TreeNode& node, int16_t* values) {
    if (node.depth == 0) {
      for (int i = 0; i < node.boards.size(); i++) {
        evaluateBoard(node.boards[i], values, i);
      }
      return;
    }
    TreeNode newNode;
    // std::vector<Move> moves;
    std::vector<uint32_t> moveToBoardIdx;
    std::vector<uint32_t> moveOffsets;
    // uint32_t movesCount = 0;
    newNode.isWhite = !node.isWhite;
    newNode.depth = node.depth - 1;
    {
      moveOffsets.resize(node.boards.size() + 1);
      std::vector<Bitboard> validMoves(node.boards.size());
      std::vector<Bitboard> attackedByEnemy(node.boards.size());
      if (node.isWhite) {
        for (int i = 0; i < node.boards.size(); i++) {
          moveOffsets[i + 1] = countWhiteMoves(node.boards[i], validMoves[i],
                                               attackedByEnemy[i]);
          if (moveOffsets[i + 1] == 0) {
            node.values = std::vector<int16_t>(node.boards.size(), 0);
            node.values[i] = MATE + node.depth;
            return;
          }
        }
      } else {
        for (int i = 0; i < node.boards.size(); i++) {
          moveOffsets[i + 1] = countBlackMoves(node.boards[i], validMoves[i],
                                               attackedByEnemy[i]);
          if (moveOffsets[i + 1] == 0) {
            node.values = std::vector<int16_t>(node.boards.size(), 0);
            node.values[i] = MATE + node.depth;
            return;
          }
        }
      }
      prefixSum(moveOffsets.data(), moveOffsets.size());
      node.movesCount = moveOffsets.back();
      node.moves.resize(node.movesCount);
      moveToBoardIdx.resize(node.movesCount);
      if (node.isWhite) {
        for (int i = 0; i < node.boards.size(); i++) {
          generateWhiteMoves(node.boards[i], validMoves[i], attackedByEnemy[i],
                             node.moves.data(), moveOffsets[i]);
          for (int j = moveOffsets[i]; j < moveOffsets[i + 1]; j++) {
            moveToBoardIdx[j] = i;
          }
        }
      } else {
        for (int i = 0; i < node.boards.size(); i++) {
          generateBlackMoves(node.boards[i], validMoves[i], attackedByEnemy[i],
                             node.moves.data(), moveOffsets[i]);
          for (int j = moveOffsets[i]; j < moveOffsets[i + 1]; j++) {
            moveToBoardIdx[j] = i;
          }
        }
      }
    }
    node.values.resize(node.movesCount);
    uint32_t movesProcessed = 0;
    while (movesProcessed < node.movesCount) {
      uint32_t movesToProcess =
          std::min(node.movesCount - movesProcessed, m_maxProcessedSize);
      newNode.boards.resize(movesToProcess);
      for (int i = 0; i < movesToProcess; i++) {
        newNode.boards[i] = node.boards[moveToBoardIdx[movesProcessed + i]];
        makeMove(newNode.boards[i], node.moves[movesProcessed + i]);
      }
      SearchNode(newNode, node.values.data() + movesProcessed);
      movesProcessed += movesToProcess;
    }
    positionsSearched[node.depth - 1] += node.movesCount;
    int16_t multiplier = node.isWhite ? 1 : -1;
    for (int i = 0; i < node.boards.size(); i++) {
      int16_t maxVal = -32768;
      for (int j = moveOffsets[i]; j < moveOffsets[i + 1]; j++) {
        if (multiplier * node.values[i] > maxVal) {
          maxVal = multiplier * node.values[i];
        }
      }
      values[i] = maxVal;
    }
    return;
  }

 private:
  Board m_startingBoard;
  bool m_startingIsWhite;
  /* Board* m_currentBoards;
   uint32_t boardsLength;*/
  Bitboard* m_validMoves;
  Bitboard* m_attackedByEnemy;
  // Board* m_nextBoards;
  uint32_t m_maxProcessedSize = 1000;
  uint32_t m_maxDepth;
  std::vector<uint32_t> positionsSearched;
};
}  // namespace engine
}  // namespace shogi
