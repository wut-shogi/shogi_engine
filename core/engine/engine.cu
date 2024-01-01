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
  /*SEARCH::init();
  std::cout << sizeof(Board) << std::endl;
  Board startingBoard = Boards::STARTING_BOARD();
  print_Board(startingBoard);
  bool isWhite = false;

  Board board = startingBoard;
  print_Board(board);
  Move bestMoveCPU = SEARCH::GetBestMove(board, false, 6, 0, SEARCH::CPU);
  std::cout << moveToUSI(bestMoveCPU) << std::endl;
  Move bestMoveGPU = SEARCH::GetBestMove(board, false, 6, 0, SEARCH::GPU);
  std::cout << moveToUSI(bestMoveGPU) << std::endl;
  std::cout << "Done!" << std::endl;*/
  GameSimulator simulator(5, 3000, SEARCH::GPU);
  simulator.Run();
}

}  // namespace engine
}  // namespace shogi
