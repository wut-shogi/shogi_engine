#include "Board.h"

Board Boards::STARTING_BOARD() {
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
      Bitboards::STARTING_ALL_BLACK(),
      Rotate90Clockwise(Bitboards::STARTING_ALL_WHITE() |
                        Bitboards::STARTING_ALL_BLACK()),
      Rotate45Clockwise(Bitboards::STARTING_ALL_WHITE() |
                        Bitboards::STARTING_ALL_BLACK()),
      Rotate45AntiClockwise(Bitboards::STARTING_ALL_WHITE() |
                            Bitboards::STARTING_ALL_BLACK()),
  }};
  return b;
}

template <>
const Bitboard& GetPlayerPieces<Player::Type::WHITE>(const Board& position) {
  return position[BB::Type::ALL_WHITE];
}

template <>
const Bitboard& GetPlayerPieces<Player::Type::BLACK>(const Board& position) {
  return position[BB::Type::ALL_BLACK];
}

template <>
Bitboard GetPieces<Piece::Type::PROMOTED_PAWN>(const Board& position) {
  return position[BB::Type::PAWN] & position[BB::Type::PROMOTED];
}

template <>
Bitboard GetPieces<Piece::Type::PROMOTED_KNIGHT>(const Board& position) {
  return position[BB::Type::KNIGHT] &
         position[BB::Type::PROMOTED];
}

template <>
Bitboard GetPieces<Piece::Type::PROMOTED_SILVER_GENERAL>(
    const Board& position) {
  return position[BB::Type::SILVER_GENERAL] &
         position[BB::Type::PROMOTED];
}

template <>
Bitboard GetPieces<Piece::Type::PROMOTED_LANCE>(const Board& position) {
  return position[BB::Type::LANCE] & position[BB::Type::PROMOTED];
}

template <>
Bitboard GetPieces<Piece::Type::HORSE>(const Board& position) {
  return position[BB::Type::BISHOP] &
         position[BB::Type::PROMOTED];
}

template <>
Bitboard GetPieces<Piece::Type::DRAGON>(const Board& position) {
  return position[BB::Type::ROOK] & position[BB::Type::PROMOTED];
}

Bitboard GetPiecesDynamic(Piece::Type piece, const Board& position) {
  switch (piece) {
    case Piece::PAWN:
      return GetPieces<Piece::Type::PAWN>(position);
    case Piece::KNIGHT:
      return GetPieces<Piece::Type::KNIGHT>(position);
    case Piece::SILVER_GENERAL:
      return GetPieces<Piece::Type::SILVER_GENERAL>(position);
    case Piece::GOLD_GENERAL:
      return GetPieces<Piece::Type::GOLD_GENERAL>(position);
    case Piece::KING:
      return GetPieces<Piece::Type::KING>(position);
    case Piece::LANCE:
      return GetPieces<Piece::Type::LANCE>(position);
    case Piece::BISHOP:
      return GetPieces<Piece::Type::BISHOP>(position);
    case Piece::ROOK:
      return GetPieces<Piece::Type::ROOK>(position);
    case Piece::HORSE:
      return GetPieces<Piece::Type::HORSE>(position);
    case Piece::DRAGON:
      return GetPieces<Piece::Type::DRAGON>(position);
    default:
      return {0, 0, 0};
  }
}

Bitboard GetPlayerPiecesDynamic(Piece::Type piece,
                                Player::Type player,
                                const Board& position) {
  switch (player) {
    case Player::Type::WHITE:
      switch (piece) {
        case Piece::PAWN:
          return GetPlayerPieces<Piece::Type::PAWN, Player::Type::WHITE>(
              position);
        case Piece::KNIGHT:
          return GetPlayerPieces<Piece::Type::KNIGHT, Player::Type::WHITE>(
              position);
        case Piece::SILVER_GENERAL:
          return GetPlayerPieces<Piece::Type::SILVER_GENERAL,
                                 Player::Type::WHITE>(position);
        case Piece::GOLD_GENERAL:
          return GetPlayerPieces<Piece::Type::GOLD_GENERAL,
                                 Player::Type::WHITE>(position);
        case Piece::KING:
          return GetPlayerPieces<Piece::Type::KING, Player::Type::WHITE>(
              position);
        case Piece::LANCE:
          return GetPlayerPieces<Piece::Type::LANCE, Player::Type::WHITE>(
              position);
        case Piece::BISHOP:
          return GetPlayerPieces<Piece::Type::BISHOP, Player::Type::WHITE>(
              position);
        case Piece::ROOK:
          return GetPlayerPieces<Piece::Type::ROOK, Player::Type::WHITE>(
              position);
        case Piece::HORSE:
          return GetPlayerPieces<Piece::Type::HORSE, Player::Type::WHITE>(
              position);
        case Piece::DRAGON:
          return GetPlayerPieces<Piece::Type::DRAGON, Player::Type::WHITE>(
              position);
        default:
          return {0, 0, 0};
      }
    case Player::Type::BLACK:
      switch (piece) {
        case Piece::PAWN:
          return GetPlayerPieces<Piece::Type::PAWN, Player::Type::BLACK>(
              position);
        case Piece::KNIGHT:
          return GetPlayerPieces<Piece::Type::KNIGHT, Player::Type::BLACK>(
              position);
        case Piece::SILVER_GENERAL:
          return GetPlayerPieces<Piece::Type::SILVER_GENERAL,
                                 Player::Type::BLACK>(position);
        case Piece::GOLD_GENERAL:
          return GetPlayerPieces<Piece::Type::GOLD_GENERAL,
                                 Player::Type::BLACK>(position);
        case Piece::KING:
          return GetPlayerPieces<Piece::Type::KING, Player::Type::BLACK>(
              position);
        case Piece::LANCE:
          return GetPlayerPieces<Piece::Type::LANCE, Player::Type::BLACK>(
              position);
        case Piece::BISHOP:
          return GetPlayerPieces<Piece::Type::BISHOP, Player::Type::BLACK>(
              position);
        case Piece::ROOK:
          return GetPlayerPieces<Piece::Type::ROOK, Player::Type::BLACK>(
              position);
        case Piece::HORSE:
          return GetPlayerPieces<Piece::Type::HORSE, Player::Type::BLACK>(
              position);
        case Piece::DRAGON:
          return GetPlayerPieces<Piece::Type::DRAGON, Player::Type::BLACK>(
              position);
        default:
          return {0, 0, 0};
      }
  }
}

template <>
uint16_t GetNumberOfPiecesInHand<Piece::Type::PAWN, Player::Type::WHITE>(
    const Board& position) {
  return position.inHandPieces.White.Pawn;
}
template <>
uint16_t GetNumberOfPiecesInHand<Piece::Type::LANCE, Player::Type::WHITE>(
    const Board& position) {
  return position.inHandPieces.White.Lance;
}
template <>
uint16_t GetNumberOfPiecesInHand<Piece::Type::KNIGHT, Player::Type::WHITE>(
    const Board& position) {
  return position.inHandPieces.White.Knight;
}
template <>
uint16_t GetNumberOfPiecesInHand<Piece::Type::SILVER_GENERAL, Player::Type::WHITE>(
    const Board& position) {
  return position.inHandPieces.White.SilverGeneral;
}
template <>
uint16_t GetNumberOfPiecesInHand<Piece::Type::GOLD_GENERAL, Player::Type::WHITE>(
    const Board& position) {
  return position.inHandPieces.White.GoldGeneral;
}
template <>
uint16_t GetNumberOfPiecesInHand<Piece::Type::BISHOP, Player::Type::WHITE>(
    const Board& position) {
  return position.inHandPieces.White.Bishop;
}
template <>
uint16_t GetNumberOfPiecesInHand<Piece::Type::ROOK, Player::Type::WHITE>(
    const Board& position) {
  return position.inHandPieces.White.Rook;
}

template <>
uint16_t GetNumberOfPiecesInHand<Piece::Type::PAWN, Player::Type::BLACK>(
    const Board& position) {
  return position.inHandPieces.Black.Pawn;
}
template <>
uint16_t GetNumberOfPiecesInHand<Piece::Type::LANCE, Player::Type::BLACK>(
    const Board& position) {
  return position.inHandPieces.Black.Lance;
}
template <>
uint16_t GetNumberOfPiecesInHand<Piece::Type::KNIGHT, Player::Type::BLACK>(
    const Board& position) {
  return position.inHandPieces.Black.Knight;
}
template <>
uint16_t GetNumberOfPiecesInHand<Piece::Type::SILVER_GENERAL,
                                 Player::Type::BLACK>(const Board& position) {
  return position.inHandPieces.Black.SilverGeneral;
}
template <>
uint16_t GetNumberOfPiecesInHand<Piece::Type::GOLD_GENERAL,
                                 Player::Type::BLACK>(const Board& position) {
  return position.inHandPieces.Black.GoldGeneral;
}
template <>
uint16_t GetNumberOfPiecesInHand<Piece::Type::BISHOP, Player::Type::BLACK>(
    const Board& position) {
  return position.inHandPieces.Black.Bishop;
}
template <>
uint16_t GetNumberOfPiecesInHand<Piece::Type::ROOK, Player::Type::BLACK>(
    const Board& position) {
  return position.inHandPieces.Black.Rook;
}