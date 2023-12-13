#include <iostream>
#include "GameTree.h"
#include "engine.h"
#include "MoveGen.h"

namespace shogi {
namespace engine {

void test() {
  std::cout << "Test!\n";
  std::cout << sizeof(Move) << std::endl;
  std::cout << sizeof(Board) << std::endl;
  std::cout << sizeof(Bitboard) << std::endl;
  std::cout << sizeof(InHandLayout) << std::endl;
  Board startingBoard = Boards::STARTING_BOARD();
  print_Board(startingBoard);
  bool isWhite;
  std::string final =
      "lnsgkgsnl/1r5+B1/pppppp1pp/6p2/9/2P6/PP1PPPPPP/7R1/LNSGKGSNL w B 4";
  Board board = Board::FromSFEN(
      "lnsgkgsnl/1r7/pppppp1pp/6p2/9/2P6/PP1PPPPPP/1+b5R1/LNSGKGSNL b b 4",
      isWhite);
  print_Board(board);
  Move move;
  move.promotion = 0;
  move.from = 74;
  move.to = 64;
  uint32_t id = 0;
  Board after;
  /*td::vector<uint32_t> moveCount(2);
  Bitboard validMoves, attackedByEnemy;
  CPU::countWhiteMoves(&board, 1, &validMoves, &attackedByEnemy,
                       moveCount.data() + 1);
  std::vector<Move> outMoves(moveCount.back());
  std::vector<uint32_t> outMoveToBoardIDx(moveCount.back());
  CPU::generateWhiteMoves(&board, 1, &validMoves, &attackedByEnemy,
                          moveCount.data(), outMoves.data(),
                          outMoveToBoardIDx.data());*/
  // benchmarkTreeBuilding(5);
  // board = startingBoard;
  GameTree tree(startingBoard, true, 3);
  Move bestMove = tree.FindBestMove();
  std::cout << "Best found move (from, to, promotion): (" << bestMove.from
            << ", " << bestMove.to << ", " << bestMove.promotion
            << std::endl;
}

}  // namespace engine
}  // namespace shogi
