#include "GameTree.h"
#include <chrono>

namespace shogi {
namespace engine {

void prefixSum(uint32_t* array, int length) {
  array[0] = 0;
  for (int i = 1; i < length; i++) {
    array[i] += array[i - 1];
  }
}

TreeLevel buildNextLevel(TreeLevel& level) {
  TreeLevel nextLevel;
  std::vector<uint32_t> moveCounts(level.length + 1);
  std::vector<Bitboard> validMoves(level.length);
  std::vector<Bitboard> attackedByEnemy(level.length);
  if (!level.isWhite) {
    for (int i = 0; i < level.length; i++) {
      moveCounts[i + 1] = countWhiteMoves(level.boardsArray[i], validMoves[i],
                                          attackedByEnemy[i]);
    }
    prefixSum(moveCounts.data(), moveCounts.size());
    nextLevel.length = moveCounts.back();
    std::cout << "Length: " << nextLevel.length << std::endl;
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

void benchmarkTreeBuilding(int maxDepth) {
  Board startingBoard = Boards::STARTING_BOARD();
  std::vector<TreeLevel> levels(maxDepth + 1);
  levels[0].depth = 0;
  levels[0].length = 1;
  levels[0].boardsArray = new Board();
  levels[0].boardsArray[0] = startingBoard;
  levels[0].isWhite = false;

  for (int i = 0; i < maxDepth; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    levels[i + 1] = buildNextLevel(levels[i]);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << levels[i + 1].length << " nodes in " << duration << " ms"
              << std::endl;
  }
  std::cout << "Done!" << std::endl;
}
}  // namespace engine
}  // namespace shogi
