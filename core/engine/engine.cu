#include <iostream>
#include "../include/engine.h"
#include "CPUsearchHelpers.h"
#include "game.h"
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

  board = Board::FromSFEN(
      "lnsgkgsn1/9/pppp1pppp/4p2+B1/B1R1L3b/4l1r2/PPPPPP1PP/9/1NSGKGSNL b",
      isWhite);

  
  std::vector<Move> movesFromRoot;
  search::init();
  // CPU::perft<true>(board, 5, movesFromRoot, isWhite);
  search::GetBestMove(board, isWhite, 0, 5);
  search::cleanup();
  std::cout << "Done!" << std::endl;
}

}  // namespace engine
}  // namespace shogi
