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
  Bitboard all =
      startingBoard[BB::Type::ALL_WHITE] | startingBoard[BB::Type::ALL_BLACK];
  std::cout << "Move count white: " << countAllMoves(startingBoard, true) << std::endl;
  std::cout << "Move count black: " << countAllMoves(startingBoard, false) << std::endl;
}


}  // namespace engine
}  // namespace shogi
