#include <iostream>
#include "../include/engine.h"
#include "CPUsearchHelpers.h"
#include "MoveGenHelpers.h"
#include "game.h"
#include "lookUpTables.h"
#include "search.h"
namespace shogi {
namespace engine {

void test() {
  Board startingBoard = Boards::STARTING_BOARD();
  bool isWhite = false;
  GameSimulator simulator(5, 3000, SEARCH::GPU);
  simulator.Run();
}

}  // namespace engine
}  // namespace shogi
