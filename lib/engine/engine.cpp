#include <iostream>
#include "Board.h"
#include "MoveGen.h"

namespace shogi {
namespace engine {

void test2() {
  std::cout << "Test!\n";
}

void test() {
  std::cout << "Test!\n";

  Board startingBoard = Boards::STARTING_BOARD();
  bool isWhite;
  Board board = Board::FromSFEN(
      "lnsgkgsn1/9/pppp1pppp/4p2+B1/B1R1L3b/4l1r2/PPPPPP1PP/9/1NSGKGSNL b", isWhite);
  auto moves = getAllLegalMoves(board, isWhite);
  std::cout << "Done" << std::endl;
  /*std::cout << "Move count white: " << countAllMoves(startingBoard, true) << std::endl;
  std::cout << "Move count black: " << countAllMoves(startingBoard, false) << std::endl;
  Board board = Board::FromSFEN(
      "lnsgkgsn1/9/pppp1pppp/4p2+B1/B1R1L3b/4l1r2/PPPPPPPPP/9/1NSGKGSNL b 1");
  print_Board(board);
  std::cout << std::endl;
  Bitboard pinned = getWhitePinnedPieces(board);
  print_BB(pinned);
  std::cout << std::endl;
  pinned = getBlackPinnedPieces(board);
  print_BB(pinned);*/
}


}  // namespace engine
}  // namespace shogi
