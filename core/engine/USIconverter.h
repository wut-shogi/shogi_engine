#pragma once

#include "Board.h"

namespace shogi {
namespace engine {
Board SFENToBoard(const std::string& boardSFEN, bool& isWhite);
std::string BoardToSFEN(const Board& board, bool isWhite);
Move USIToMove(const std::string& USImove);
std::string MoveToUSI(Move move);

std::vector<std::string> boardToStringVector(const Board& board);
std::string inHandToString(const InHandLayout& inHand);
}  // namespace engine
}  // namespace shogi