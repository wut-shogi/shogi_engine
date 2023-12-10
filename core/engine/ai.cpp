#include "ai.h"

namespace shogi {
namespace engine {

void prefixSum(int* array, int length) {
  array[0] = 0;
  for (int i = 1; i < length; i++) {
    array[i] += array[i-1];
  }
}

TreeLevel buildNextLevel(TreeLevel& level) {
  TreeLevel nextLevel;
  std::vector<int> moveCounts(level.length+1);
  std::vector<Bitboard> validMoves(level.length);
  std::vector<Bitboard> attackedByEnemy(level.length);
  if (!level.isWhite) {
    for (int i = 0; i < level.length; i++) {
      moveCounts[i + 1] = countWhiteMoves(level.boardsArray[i], validMoves[i],
                                          attackedByEnemy[i]);
    }
    prefixSum(moveCounts.data(), moveCounts.size());
    nextLevel.length = moveCounts.back();
    nextLevel.movesArray = new Move[nextLevel.length];
    nextLevel.boardsArray = new Board[nextLevel.length];
    for (int i = 0; i < level.length; i++) {
      generateWhiteMoves(level.boardsArray[i], validMoves[i],
                         attackedByEnemy[i], nextLevel.movesArray,
                         moveCounts[i]);
    }
    for (int i = 0; i < level.length; i++) {
      generateNextBoards(level.boardsArray[i],
                         nextLevel.movesArray + moveCounts[i],
                         moveCounts[i + 1] - moveCounts[i],
                         nextLevel.boardsArray + moveCounts[i]);
    }
  } else {
    for (int i = 0; i < level.length; i++) {
      moveCounts[i + 1] = countBlackMoves(level.boardsArray[i], validMoves[i],
                                          attackedByEnemy[i]);
    }
    prefixSum(moveCounts.data(), moveCounts.size());
    nextLevel.length = moveCounts.back();
    nextLevel.movesArray = new Move[nextLevel.length];
    nextLevel.boardsArray = new Board[nextLevel.length];
    for (int i = 0; i < level.length; i++) {
      generateBlackMoves(level.boardsArray[i], validMoves[i],
                         attackedByEnemy[i], nextLevel.movesArray,
                         moveCounts[i]);
    }
    for (int i = 0; i < level.length; i++) {
      generateNextBoards(level.boardsArray[i],
                         nextLevel.movesArray + moveCounts[i],
                         moveCounts[i + 1] - moveCounts[i],
                         nextLevel.boardsArray + moveCounts[i]);
    }
  }
  nextLevel.isWhite = !level.isWhite;
  nextLevel.depth = level.depth + 1;
  // Free previous layer boards
  delete[] level.boardsArray;
  return nextLevel;
}
}  // namespace engine
}  // namespace shogi
