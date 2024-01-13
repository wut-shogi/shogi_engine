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
  GameSimulator simulator(5, 0, SEARCH::GPU);
  simulator.Run();
  /*SEARCH::init();
  SEARCH::perftGPU<true>(startingBoard, 6, isWhite);*/
  SEARCH::cleanup();
}

}  // namespace engine
}  // namespace shogi
