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
    assert(idx >= BitboardType::BEGIN && idx < BitboardType::SIZE);

    return bbs[idx];
  }

  Board& operator=(const Board& board) {
    std::memcpy(this->bbs, board.bbs, sizeof(this->bbs));
    return *this;
  }
};

namespace Boards {
Board STARTING_BOARD();
}
