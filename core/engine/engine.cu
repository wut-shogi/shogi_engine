#include <iostream>
#include "../include/engine.h"
#include "CPUsearchHelpers.h"
#include "game.h"
#include "lookUpTables.h"
#include "moveGenHelpers.h"
#include "search.h"
namespace shogi {
namespace engine {

void test() {
  Board startingBoard = Boards::STARTING_BOARD();
  print_Board(startingBoard);
  bool isWhite = false;

  /*Board board;
  board[BB::Type::PAWN] = {0, 133955584, 133955584};
  board[BB::Type::PAWN] = {67371008, 0, 257};
  board[BB::Type::PAWN] = {34078720, 0, 130};
  board[BB::Type::PAWN] = {34816, 0, 68};
  board[BB::Type::PAWN] = {10485760, 0, 40};
  board[BB::Type::PAWN] = {1024, 0, 65536};
  board[BB::Type::PAWN] = {65536, 0, 1024};
  board[BB::Type::PAWN] = {4096, 0, 4096};
  board[BB::Type::PAWN] = {0, 0, 0};
  board[BB::Type::PAWN] = {112040960, 133955584, 0};
  board[BB::Type::PAWN] = {0, 0, 134026735};
  print_Board(board);*/

  std::vector<Move> movesFromRoot;
  //SEARCH::init();
  // CPU::perft<true, true>(startingBoard, 4, movesFromRoot, false);
  //Move bestMove = SEARCH::GetBestMove(startingBoard, isWhite, 5);
  //Move bestMoveCPU = SEARCH::GetBestMoveAlphaBeta(startingBoard, isWhite, 5);
  // search::cleanup();
  GameSimulator simulator({SEARCH::GetBestMoveAlphaBeta, SEARCH::GetBestMove});
  simulator.Run();
  std::cout << "Done!" << std::endl;
}

}  // namespace engine
}  // namespace shogi
