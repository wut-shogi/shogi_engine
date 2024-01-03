#include <iostream>
#include "../include/engine.h"
#include "CPUsearchHelpers.h"
#include "lookUpTables.h"
#include "moveGenHelpers.h"
#include "search.h"
#include "game.h"
namespace shogi {
namespace engine {

void test() {
  Board startingBoard = Boards::STARTING_BOARD();
  bool isWhite = false;
  GameSimulator simulator(6, 3000, SEARCH::GPU);
  simulator.Run();
}

}  // namespace engine
}  // namespace shogi
