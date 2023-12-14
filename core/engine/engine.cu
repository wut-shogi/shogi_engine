#include <iostream>
#include "GameTree.h"
#include "engine.h"
#include "MoveGen.h"
#include "cpuInterface.h"

namespace shogi {
namespace engine {

void test() {
  Board startingBoard = Boards::STARTING_BOARD();
  bool isWhite;
  GameTree tree(startingBoard, isWhite, 3);
  Move bestMove = tree.FindBestMove();
  std::cout << "Best found move (from, to, promotion): (" << bestMove.from
            << ", " << bestMove.to << ", " << bestMove.promotion << ")"
            << std::endl;
}

}  // namespace engine
}  // namespace shogi
