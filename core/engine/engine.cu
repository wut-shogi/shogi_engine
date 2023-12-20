#include "../include/engine.h"
#include <iostream>
#include "CPUsearchHelpers.h"
#include "lookUpTables.h"
#include "moveGenHelpers.h"
namespace shogi {
namespace engine {

void test() {
  Board startingBoard = Boards::STARTING_BOARD();
  print_Board(startingBoard);
  bool isWhite = true;
  Board board = Board::FromSFEN(
      "1R3G1nl/4g1kg1/1p2p+bpp1/p+B1Ls1s1p/Pn3+l3/KSPpP3P/3P+rpP2/2G6/L8 b "
      "NPsn3p 1",
      isWhite);
  /*Move move;
  move.from = 55;
  move.to = 46;
  move.promotion = true;
  makeMove(startingBoard, move);
  print_Board(startingBoard);
  move.from = 16;
  move.to = 46;
  move.promotion = true;
  makeMove(startingBoard, move);
  print_Board(startingBoard);*/
  /*GameTree tree(startingBoard, isWhite, 5);
  Move bestMove = tree.FindBestMove();
  std::cout << "Best found move (from, to, promotion): (" << bestMove.from
            << ", " << bestMove.to << ", " << bestMove.promotion << ")"
            << std::endl;*/

  /*std::vector<uint32_t> offsets(2);
  Move move;
  Bitboard valid, attacked, pinned;
  bool isMate = false;
  CPU::countBlackMoves((uint32_t)1, (int16_t)0, startingBoard, &move, offsets.data() + 1, &valid,
                       &attacked, &pinned, &isMate);
  CPU::prefixSum(offsets.data(), offsets.size());
  std::vector<Move> moves(offsets.back());
  CPU::generateBlackMoves(1, 0, startingBoard, &move, offsets.data(), &valid,
                          &attacked, &pinned, moves.data());*/


  // Trzymaæ ruchy w osobnych tablicach
  /*GPU::initLookUpArrays();
  Move bestMove = GetBestMove(board, isWhite, 0, 6);
  std::cout << "From: " << bestMove.from << ", To: " << bestMove.to
            << ", Promotion: " << bestMove.promotion << std::endl;*/
  /*GameSimulator simulator;
  simulator.Run();*/
  board[BB::Type::PAWN] = {511, 0, 133955584};
  board[BB::Type::LANCE] = {67371008, 0, 257};
  board[BB::Type::KNIGHT] = {34078720, 0, 130};
  board[BB::Type::SILVER_GENERAL] = {17825792, 0, 68};
  board[BB::Type::GOLD_GENERAL] = {10485760, 0, 40};
  board[BB::Type::BISHOP] = {1024, 0, 65536};
  board[BB::Type::ROOK] = {65536, 0, 1024};
  board[BB::Type::KING] = {4096, 0, 2048};
  board[BB::Type::PROMOTED] = {0, 0, 0};
  board[BB::Type::ALL_WHITE] = {129832447, 0, 2048};
  board[BB::Type::ALL_BLACK] = {0, 0, 134022655};
  print_Board(board);

  LookUpTables::CPU::init();
  //CPU::MoveList moves(board, true);
  /*Bitboard empty;
  Bitboard result = LookUpTables::getDiagRightAttacks(
      H8, startingBoard[BB::Type::ALL_WHITE] | startingBoard[BB::Type::ALL_BLACK]);
  print_BB(result);*/
  Move move;
  move.from = 56;
  move.to = 47;
  move.promotion = 0;
  makeMove(startingBoard, move);
  move.from = 5;
  move.to = 14;
  makeMove(startingBoard, move);
  move.from = H8;
  move.to = C3;
  makeMove(startingBoard, move);
  /* move.from = 26;
  move.to = 35;
  makeMove(startingBoard, move);
  move.from = 64;
  move.to = 48;
  makeMove(startingBoard, move);
  move.from = 4;
  move.to = 14;
  makeMove(startingBoard, move);
  move.from = 48;
  move.to = 24;
  move.promotion = 1;
  makeMove(startingBoard, move);*/
  print_Board(startingBoard);
  CPU::perft<true>(startingBoard, 2, true);
  std::cout << "Done!" << std::endl;
}

}  // namespace engine
}  // namespace shogi
