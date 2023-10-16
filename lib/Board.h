#pragma once
#include <cassert>
#include "Bitboard.h"
#include "Rules.h"
struct Board {
  Bitboard bbs[BitboardType::SIZE];

  Board(std::array<Bitboard, BitboardType::SIZE>&& bbs) {
    std::memcpy(this->bbs, bbs.data(), sizeof(this->bbs));
  }

  Bitboard& operator[](BitboardType idx) {
    assert(idx >= BitboardType::BEGIN && idx < Bitboardtype::SIZE);

    return bbs[idx];
  }

  Board& operator=(const Board& board) {
    std::memcpy(this->bbs, board.bbs, sizeof(this->bbs));
    return *this;
  }
};

void InitializeWithStartingPosition(Board& board) {
  board = Boards::startingBoard;
}

namespace Boards {
static Board startingBoard;
}