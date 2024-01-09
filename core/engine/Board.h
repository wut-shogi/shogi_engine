#pragma once
#include <cassert>
#include <sstream>
#include "Bitboard.h"
#include "Rules.h"
namespace shogi {
namespace engine {
struct Board {
  Bitboard bbs[BB::Type::SIZE];
  InHandLayout inHand;
  RUNTYPE Board() {}

  Board(std::array<Bitboard, BB::Type::SIZE>&& bbs, InHandLayout inHand)
      : inHand(inHand) {
    std::memcpy(this->bbs, bbs.data(), sizeof(this->bbs));
  }

  RUNTYPE Bitboard& operator[](BB::Type idx) { return bbs[idx]; }

  RUNTYPE const Bitboard& operator[](BB::Type idx) const {
    return bbs[idx];
  }
};

void print_Board(const Board& board);

namespace Boards {
Board STARTING_BOARD();
}
}  // namespace engine
}  // namespace shogi
