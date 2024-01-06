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

  Board board = SFENToBoard(
      "lnsg1gsnl/1r5b1/ppp2pppp/4p4/5k3/P1PB5/NP1PPPPPP/4K2R1/L1SG1GSNL b P 1",
      isWhite);
  print_Board(board);
  Move bestMoveCPU = SEARCH::GetBestMove(board, false, 5, 0, SEARCH::CPU);
  std::cout << MoveToUSI(bestMoveCPU) << std::endl;
  Move bestMoveGPU = SEARCH::GetBestMove(board, false,5, 0, SEARCH::GPU);
  std::cout << MoveToUSI(bestMoveGPU) << std::endl;
  std::cout << "Done!" << std::endl;
  //SEARCH::perftCPU<true>(startingBoard, 3, false);
  /*GameSimulator simulator(5, 3000, SEARCH::GPU);
  simulator.Run();*/
}

}  // namespace engine
}  // namespace shogi
