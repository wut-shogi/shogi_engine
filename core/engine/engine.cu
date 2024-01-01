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
  SEARCH::init();
  std::cout << sizeof(Board) << std::endl;
  Board startingBoard = Boards::STARTING_BOARD();
  print_Board(startingBoard);
  bool isWhite = false;

  Board board = startingBoard;
  /*board[BB::Type::PAWN] = {506, 262208, 117178368};
  board[BB::Type::LANCE] = {67371008, 0, 257};
  board[BB::Type::KNIGHT] = {34078720, 0, 130};
  board[BB::Type::SILVER_GENERAL] = {16781312, 0, 68};
  board[BB::Type::GOLD_GENERAL] = {10485760, 0, 40};
  board[BB::Type::BISHOP] = {1028, 0, 0};
  board[BB::Type::ROOK] = {65536, 0, 1024};
  board[BB::Type::KING] = {4194304, 0, 4096};
  board[BB::Type::PROMOTED] = {0, 0, 0};
  board[BB::Type::ALL_WHITE] = {132978170, 262144, 0};
  board[BB::Type::ALL_BLACK] = {4, 64, 117183983};
  board.inHand.value = 268435456;*/
  //print_Board(board);
 // std::cout << boardToSFEN(board);
  //CPU::MoveList moves(board, true);
  /* auto start = std::chrono::high_resolution_clock::now();
   Move bestMove = SEARCH::GetBestMove2(board, isWhite, 4);
   auto stop = std::chrono::high_resolution_clock::now();
   auto duration =
       std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
   std::cout << "Time: " << duration.count() << " ms" << std::endl;
   std::cout << moveToUSI(bestMove) << std::endl;*/
  /*Move move;
  move.from = G7;
  move.to = F7;
  move.promotion = 0;
  makeMove(board, move);
   move.from = C1;
  move.to = D1;
  move.promotion = 0;
  makeMove(board, move);
  move.from = H8;
  move.to = E5;
  move.promotion = 0;
  makeMove(board, move);
  move.from = A6;
  move.to = B6;
  move.promotion = 0;
  makeMove(board, move);
  move.from = E5;
  move.to = C7;
  move.promotion = 0;
  makeMove(board, move);*/

  //CPU::MoveList moves(board, true);

  print_Board(board);
  std::cout << boardToSFEN(board) << std::endl;
  SEARCH::perftCPU<true>(board, 6, false);
  SEARCH::perftGPU<true>(board, 6, false);
  //Move bestMove1 = SEARCH::GetBestMoveAlphaBeta(board, false, 6);
  //Move bestMove = SEARCH::GetBestMove2(board, false, 6);
  /* start = std::chrono::high_resolution_clock::now();
  Move bestMoveCPU = SEARCH::GetBestMoveAlphaBeta(board, isWhite, 4);
   stop = std::chrono::high_resolution_clock::now();
   duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "Time: " << duration.count() << " ms" << std::endl;
  std::cout << moveToUSI(bestMoveCPU) << std::endl;*/
  // search::cleanup();
  // GameSimulator simulator({SEARCH::GetBestMoveAlphaBeta,
  // SEARCH::GetBestMove}); simulator.Run();
  std::cout << "Done!" << std::endl;
}

}  // namespace engine
}  // namespace shogi
