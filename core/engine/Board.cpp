#include "Board.h"
#include <vector>

namespace shogi {
namespace engine {
Board Boards::STARTING_BOARD() {
  InHandLayout inHand;
  inHand.value = 0;
  static Board b = {{
                        Bitboards::STARTING_PAWN(),
                        Bitboards::STARTING_KNIGHT(),
                        Bitboards::STARTING_SILVER_GENERAL(),
                        Bitboards::STARTING_GOLD_GENERAL(),
                        Bitboards::STARTING_KING(),
                        Bitboards::STARTING_LANCE(),
                        Bitboards::STARTING_BISHOP(),
                        Bitboards::STARTING_ROOK(),
                        Bitboards::STARTING_PROMOTED(),
                        Bitboards::STARTING_ALL_WHITE(),
                        Bitboards::STARTING_ALL_BLACK()
                    },
      inHand};
  return b;
}

std::vector<std::string> boardToStringVector(const Board& board) {
  std::vector<std::string> boardRepresentation(BOARD_SIZE);
  Bitboard promoted = board[BB::Type::PROMOTED];
  Bitboard notPromoted = ~promoted;
  Bitboard playerMask;
  BitboardIterator iterator;

  // White
  playerMask = board[BB::Type::ALL_WHITE];
  // Pawns
  iterator.Init(board[BB::Type::PAWN] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "p";
  }
  iterator.Init(board[BB::Type::PAWN] & playerMask & promoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "+p";
  }
  // Lances
  iterator.Init(board[BB::Type::LANCE] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "l";
  }
  iterator.Init(board[BB::Type::LANCE] & playerMask & promoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "+l";
  }
  // Knight
  iterator.Init(board[BB::Type::KNIGHT] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "n";
  }
  iterator.Init(board[BB::Type::KNIGHT] & playerMask & promoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "+n";
  }
  // Silver Generals
  iterator.Init(board[BB::Type::SILVER_GENERAL] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "s";
  }
  iterator.Init(board[BB::Type::SILVER_GENERAL] & playerMask & promoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "+s";
  }
  // Gold Generals
  iterator.Init(board[BB::Type::GOLD_GENERAL] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "g";
  }
  // Bishops
  iterator.Init(board[BB::Type::BISHOP] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "b";
  }
  iterator.Init(board[BB::Type::BISHOP] & playerMask & promoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "+b";
  }
  // Rooks
  iterator.Init(board[BB::Type::ROOK] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "r";
  }
  iterator.Init(board[BB::Type::ROOK] & playerMask & promoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "+r";
  }
  // Kings
  iterator.Init(board[BB::Type::KING] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "k";
  }

  // Black
  playerMask = board[BB::Type::ALL_BLACK];
  // Pawns
  iterator.Init(board[BB::Type::PAWN] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "P";
  }
  iterator.Init(board[BB::Type::PAWN] & playerMask & promoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "+P";
  }
  // Lances
  iterator.Init(board[BB::Type::LANCE] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "L";
  }
  iterator.Init(board[BB::Type::LANCE] & playerMask & promoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "+L";
  }
  // Knight
  iterator.Init(board[BB::Type::KNIGHT] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "N";
  }
  iterator.Init(board[BB::Type::KNIGHT] & playerMask & promoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "+N";
  }
  // Silver Generals
  iterator.Init(board[BB::Type::SILVER_GENERAL] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "S";
  }
  iterator.Init(board[BB::Type::SILVER_GENERAL] & playerMask & promoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "+S";
  }
  // Gold Generals
  iterator.Init(board[BB::Type::GOLD_GENERAL] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "G";
  }
  // Bishops
  iterator.Init(board[BB::Type::BISHOP] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "B";
  }
  iterator.Init(board[BB::Type::BISHOP] & playerMask & promoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "+B";
  }
  // Rooks
  iterator.Init(board[BB::Type::ROOK] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "R";
  }
  iterator.Init(board[BB::Type::ROOK] & playerMask & promoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "+R";
  }
  // Kings
  iterator.Init(board[BB::Type::KING] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "K";
  }

  return boardRepresentation;
}

void print_Board(const Board& board) {
  std::vector<std::string> boardRepresentation = boardToStringVector(board);
  for (int i = 0; i < BOARD_SIZE; i++) {
    if (i == 0) {
      std::cout << std::endl << "|";
    }
    if (i != 0 && i % 9 == 0) {
      std::cout << std::endl;
      for (int j = 0; j < BOARD_DIM; j++) {
        std::cout << " ---";
      }
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
  std::cout << std::endl;
}

std::string boardToSFEN(const Board& board) {
  std::vector<std::string> boardRepresentation = boardToStringVector(board);
  std::string result = "";
  int number = 0;
  for (int i = 0; i < BOARD_SIZE; i++) {
    if (boardRepresentation[i].empty()) {
      number++;
    } else {
      if (number > 0) {
        result += std::to_string(number);
        number = 0;
      }
      result += boardRepresentation[i];
    }
    if ((i + 1) % BOARD_DIM == 0) {
      if (number > 0) {
        result += std::to_string(number);
        number = 0;
      }
      if (i != BOARD_SIZE - 1)
        result += "/";
    }
  }
  return result;
}
}  // namespace engine
}  // namespace shogi