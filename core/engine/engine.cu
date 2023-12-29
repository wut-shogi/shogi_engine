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
  std::cout << sizeof(Board) << std::endl;
  Board startingBoard = Boards::STARTING_BOARD();
  print_Board(startingBoard);
  bool isWhite = false;
  Board board = Board::FromSFEN(
      "ln1g3nl/4ks1+R1/p2pppp2/8p/B1s6/2P3P1P/P2PPPS2/3+p1K3/L+r3G1NL b "
      "BG2Pgsn2p 1",
      isWhite);

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
  SEARCH::init();
  auto start = std::chrono::high_resolution_clock::now();
  Move bestMove = SEARCH::GetBestMove2(board, isWhite, 4);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "Time: " << duration.count() << " ms" << std::endl;
  std::cout << moveToUSI(bestMove) << std::endl;
   //CPU::perft<true, true>(startingBoard, 6, movesFromRoot, false);
  //Move bestMove = SEARCH::GetBestMove(startingBoard, isWhite, 5);
   start = std::chrono::high_resolution_clock::now();
  Move bestMoveCPU = SEARCH::GetBestMoveAlphaBeta(board, isWhite, 4);
   stop = std::chrono::high_resolution_clock::now();
   duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "Time: " << duration.count() << " ms" << std::endl;
  std::cout << moveToUSI(bestMoveCPU) << std::endl;
  // search::cleanup();
  //GameSimulator simulator({SEARCH::GetBestMoveAlphaBeta, SEARCH::GetBestMove});
  //simulator.Run();
  std::cout << "Done!" << std::endl;
}

}  // namespace engine
}  // namespace shogi
