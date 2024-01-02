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
  __host__ __device__ Board() {}

  Board(std::array<Bitboard, BB::Type::SIZE>&& bbs, InHandLayout inHand)
      : inHand(inHand) {
    std::memcpy(this->bbs, bbs.data(), sizeof(this->bbs));
  }

  __host__ __device__ Bitboard& operator[](BB::Type idx) { return bbs[idx]; }

  __host__ __device__ const Bitboard& operator[](BB::Type idx) const {
    return bbs[idx];
  }
};

std::string boardToSFEN(const Board& board);
void print_Board(const Board& board);

namespace Boards {
Board STARTING_BOARD();
}
}  // namespace engine
}  // namespace shogi
