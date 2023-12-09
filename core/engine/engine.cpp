#include <iostream>
#include "engine.h"
#include "Board.h"
#include "MoveGen.h"

namespace shogi {
namespace engine {

void test() {
  std::cout << "Test!\n";

  Board startingBoard = Boards::STARTING_BOARD();
  bool isWhite;
  Board board = Board::FromSFEN(
      "8l/1l+R2P3/p2pBG1pp/kps1p4/Nn1P2G2/P1P1P2PP/1PS6/1KSG3+r1/LN2+p3L w Sbgn3p 124", isWhite);
  print_Board(board);
  Move bestMove = getBestMove(board, isWhite, 0, 0);
  makeMove(board, bestMove);
  print_Board(board);
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
