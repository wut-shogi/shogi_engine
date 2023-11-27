#pragma once
#include "Bitboard.h"
#include "Rules.h"
#include <cassert>
struct Board {
  Bitboard bbs[BB::Type::SIZE];

  InHand inHandPieces;

  Board(std::array<Bitboard, BB::Type::SIZE>&& bbs) {
    std::memcpy(this->bbs, bbs.data(), sizeof(this->bbs));
  }

   Bitboard& operator[](BB::Type idx) {
    return bbs[idx];
  }

   const Bitboard& operator[](BB::Type idx) const{
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

template<Player::Type player>
const Bitboard& GetPlayerPieces(const Board& position);

template<Piece::Type piece>
Bitboard GetPieces(const Board& position) {
  return position[static_cast<BB::Type>(piece)] &
         (~position[BB::Type::PROMOTED]);
}

template <>
Bitboard GetPieces<Piece::Type::PROMOTED_PAWN>(const Board& position);

template <>
Bitboard GetPieces<Piece::Type::PROMOTED_KNIGHT>(const Board& position);

template <>
Bitboard GetPieces<Piece::Type::PROMOTED_SILVER_GENERAL>(
    const Board& position);

template <>
Bitboard GetPieces<Piece::Type::PROMOTED_LANCE>(const Board& position);

template <>
Bitboard GetPieces<Piece::Type::HORSE>(const Board& position);

template <>
Bitboard GetPieces<Piece::Type::DRAGON>(const Board& position);

template <Piece::Type piece, Player::Type player>
Bitboard GetPlayerPieces(const Board& position) {
  const Bitboard& playerPieces = GetPlayerPieces<player>(position);
  const Bitboard pieces = GetPieces<piece>(position);
  Bitboard outBB = playerPieces & pieces;
  return outBB;
}

Bitboard GetPiecesDynamic(Piece::Type piece,
                          const Board& position);

Bitboard GetPlayerPiecesDynamic(Piece::Type piece,
                          Player::Type player,
                          const Board& position);

template<Piece::Type piece, Player::Type player>
uint16_t GetNumberOfPiecesInHand(const Board& position);
