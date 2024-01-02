#include <vector>
#include "Board.h"
#include "USIconverter.h"

namespace shogi {
namespace engine {
Board Boards::STARTING_BOARD() {
  InHandLayout inHand;
  inHand.value = 0;
  static Board b = {
      {Bitboards::STARTING_PAWN(), Bitboards::STARTING_LANCE(),
       Bitboards::STARTING_KNIGHT(), Bitboards::STARTING_SILVER_GENERAL(),
       Bitboards::STARTING_GOLD_GENERAL(), Bitboards::STARTING_BISHOP(),
       Bitboards::STARTING_ROOK(), Bitboards::STARTING_KING(),
       Bitboards::STARTING_PROMOTED(), Bitboards::STARTING_ALL_WHITE(),
       Bitboards::STARTING_ALL_BLACK()},
      inHand};
  return b;
}

void print_Board(const Board& board) {
  std::vector<std::string> boardRepresentation = boardToStringVector(board);
  std::cout << "| 9 | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 |" << std::endl;
  std::cout << " -----------------------------------|---" << std::endl;
  for (int i = 0; i < BOARD_SIZE; i++) {
    if (i == 0) {
      std::cout << "|";
    }
    if (i != 0 && i % 9 == 0) {
      std::cout << " " << (char)('a' + (i-1) / 9) << " ";
      std::cout << std::endl;
      for (int j = 0; j < BOARD_DIM; j++) {
        std::cout << " ---";
      }
      std::cout << "|---";
      std::cout << std::endl << "|";
    }
    if (boardRepresentation[i] == "") {
      std::cout << "   |";
      continue;
    } else if (boardRepresentation[i].size() == 1) {
      std::cout << " ";
    }
    std::cout << boardRepresentation[i] << " ";
    std::cout << "|";
  }
  std::cout << " i ";
  std::cout << std::endl;
}
}  // namespace engine
}  // namespace shogi