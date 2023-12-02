#pragma once
#include <cassert>
#include "Bitboard.h"
#include "Rules.h"
struct Board {
  Bitboard bbs[BB::Type::SIZE];
  NonBitboardPieces nonBitboardPieces;
  InHandPieces inHandPieces;

  Board(std::array<Bitboard, BB::Type::SIZE>&& bbs,
        NonBitboardPieces nonBitboardPieces,
        InHandPieces inHandPieces)
      : nonBitboardPieces(nonBitboardPieces), inHandPieces(inHandPieces) {
    std::memcpy(this->bbs, bbs.data(), sizeof(this->bbs));
  }

  Bitboard& operator[](BB::Type idx) { return bbs[idx]; }

  const Bitboard& operator[](BB::Type idx) const { return bbs[idx]; }

  Board& operator=(const Board& board) {
    std::memcpy(this->bbs, board.bbs, sizeof(this->bbs));
    return *this;
  }
};

namespace Boards {
Board STARTING_BOARD();
}

Bitboard GetPlayerPiecesDynamic(Piece::Type piece,
                                Player::Type player,
                                const Board& position);
