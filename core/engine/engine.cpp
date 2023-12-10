#include "engine.h"
#include <iostream>
#include "ai.h"

namespace shogi {
namespace engine {

void test() {
  std::cout << "Test!\n";

  Board startingBoard = Boards::STARTING_BOARD();
  int maxDepth = 5;
  std::vector<TreeLevel> levels(maxDepth + 1);
  levels[0].depth = 0;
  levels[0].length = 1;
  levels[0].boardsArray = new Board();
  levels[0].boardsArray[0] = startingBoard;
  levels[0].isWhite = false;

  for (int i = 0; i < maxDepth; i++) {
    levels[i + 1] = buildNextLevel(levels[i]);
  }
  std::cout << "Done!" << std::endl;
}

}  // namespace engine
}  // namespace shogi
