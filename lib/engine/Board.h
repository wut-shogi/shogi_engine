#pragma once
#include <cassert>
#include <sstream>
#include "Bitboard.h"
#include "Rules.h"

struct Board {
  Bitboard bbs[BB::Type::SIZE];
  InHandPieces inHandPieces;
  Board() {}

  Board(std::array<Bitboard, BB::Type::SIZE>&& bbs,
        InHandPieces inHandPieces)
      : inHandPieces(inHandPieces) {
    std::memcpy(this->bbs, bbs.data(), sizeof(this->bbs));
  }

Bitboard& operator[](BB::Type idx) {
  return bbs[idx];
}

const Bitboard& operator[](BB::Type idx) const {
  return bbs[idx];
}

Board& operator=(const Board& board) {
  std::memcpy(this->bbs, board.bbs, sizeof(this->bbs));
  return *this;
}
}
;

namespace Boards {
Board STARTING_BOARD();
}
