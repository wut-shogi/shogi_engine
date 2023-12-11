#include "engine.h"
#include <iostream>
#include "GameTree.h"

namespace shogi {
namespace engine {

void test() {
  std::cout << "Test!\n";
  std::cout << sizeof(Move) << std::endl;
  std::cout << sizeof(Board) << std::endl;
  std::cout << sizeof(Bitboard) << std::endl;
  std::cout << sizeof(InHandLayout) << std::endl;
  Board startingBoard = Boards::STARTING_BOARD();
  bool isWhite;
  Board board = Board::FromSFEN(
      "lnB5l/k1b+S5/p1p1p4/1P1g1pgpp/N8/2P1+n4/Pg1P1sp1P/3G3R1/L3KP2L w "
      "N4Pr2sp 2", isWhite);
  //benchmarkTreeBuilding(5);
  //board = startingBoard;
  GameTree tree(board, isWhite, 3);
  tree.FindBestMove();
  std::cout << "Done!" << std::endl;
}

}  // namespace engine
}  // namespace shogi
