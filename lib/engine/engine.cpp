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
  std::cout << "ffs_host: " << ffs_host(0) << std::endl;
  std::cout << "ffs_host: " << ffs_host(1) << std::endl;
  std::cout << "ffs_host: " << ffs_host(8) << std::endl;
  Board startingBoard = Boards::STARTING_BOARD();
  Bitboard all =
      startingBoard[BB::Type::ALL_WHITE] | startingBoard[BB::Type::ALL_BLACK];
  std::cout << "Move count white: " << countAllMoves(startingBoard, true) << std::endl;
  std::cout << "Move count black: " << countAllMoves(startingBoard, false) << std::endl;
}


}  // namespace engine
}  // namespace shogi
