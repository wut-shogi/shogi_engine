#pragma once
#include "Board.h"
#include "MoveGenHelpers.h"
#include <vector>

struct Move {
  Piece::Type piece;
  Square from;
  Square to;
};

size_t countAllMoves(const Board& board, bool isWhite);


std::vector<std::pair<int, int>> getLegalMovesFromSquare(std::string SFENstring,
                                                         int rank,
                                                         int file);



