#include "Board.h"

Board Boards::STARTING_BOARD() {
  InHandPieces inHandPieces;
  inHandPieces.value = 0;
  NonBitboardPieces nonBitboardPieces;
  nonBitboardPieces.White.King = Square::A5;
  nonBitboardPieces.White.Lance1 = Square::A9;
  nonBitboardPieces.White.Lance2 = Square::A1;
  nonBitboardPieces.White.Bishop = Square::B2;
  nonBitboardPieces.White.Rook = Square::B8;
  nonBitboardPieces.White.Horse = Square::NONE;
  nonBitboardPieces.White.Dragon = Square::NONE;
  nonBitboardPieces.Black.King = Square::I5;
  nonBitboardPieces.Black.Lance1 = Square::I9;
  nonBitboardPieces.Black.Lance2 = Square::I1;
  nonBitboardPieces.Black.Bishop = Square::H8;
  nonBitboardPieces.Black.Rook = Square::H2;
  nonBitboardPieces.Black.Horse = Square::NONE;
  nonBitboardPieces.Black.Dragon = Square::NONE;
  static Board b = {{
                        Bitboards::STARTING_PAWN(),
                        Bitboards::STARTING_KNIGHT(),
                        Bitboards::STARTING_SILVER_GENERAL(),
                        Bitboards::STARTING_GOLD_GENERAL(),
                        Bitboards::STARTING_PROMOTED(),
                        Bitboards::STARTING_ALL_WHITE(),
                        Bitboards::STARTING_ALL_BLACK(),
                        Rotate90Clockwise(Bitboards::STARTING_ALL_WHITE() |
                                          Bitboards::STARTING_ALL_BLACK()),
                        Rotate45Clockwise(Bitboards::STARTING_ALL_WHITE() |
                                          Bitboards::STARTING_ALL_BLACK()),
                        Rotate45AntiClockwise(Bitboards::STARTING_ALL_WHITE() |
                                              Bitboards::STARTING_ALL_BLACK()),
                    },
                    nonBitboardPieces,
                    inHandPieces};
  return b;
}

Bitboard GetPlayerPiecesDynamic(Piece::Type piece,
                                Player::Type player,
                                const Board& position) {
  Square pos, pos1, pos2;
  if (player == Player::Type::WHITE) {
    switch (piece) {
      case Piece::PAWN:
        return position.bbs[BB::Type::PAWN] & position.bbs[BB::Type::ALL_WHITE];
      case Piece::KNIGHT:
        return position.bbs[BB::Type::KNIGHT] &
               position.bbs[BB::Type::ALL_WHITE];
      case Piece::SILVER_GENERAL:
        return position.bbs[BB::Type::SILVER_GENERAL] &
               position.bbs[BB::Type::ALL_WHITE];
      case Piece::GOLD_GENERAL:
        return position.bbs[BB::Type::GOLD_GENERAL] &
               position.bbs[BB::Type::ALL_WHITE];
      case Piece::KING:
        pos = static_cast<Square>(position.nonBitboardPieces.White.King);
        return pos != Square::NONE ? Bitboard(pos) : Bitboard();
      case Piece::LANCE:
        pos1 = static_cast<Square>(position.nonBitboardPieces.White.Lance1);
        pos2 = static_cast<Square>(position.nonBitboardPieces.White.Lance2);
        return (pos1 != Square::NONE ? Bitboard(pos1) : Bitboard()) |
               (pos2 != Square::NONE ? Bitboard(pos2) : Bitboard());
      case Piece::BISHOP:
        pos = static_cast<Square>(position.nonBitboardPieces.White.Bishop);
        return pos != Square::NONE ? Bitboard(pos) : Bitboard();
      case Piece::ROOK:
        pos = static_cast<Square>(position.nonBitboardPieces.White.Rook);
        return pos != Square::NONE ? Bitboard(pos) : Bitboard();
      case Piece::HORSE:
        pos = static_cast<Square>(position.nonBitboardPieces.White.Horse);
        return pos != Square::NONE ? Bitboard(pos) : Bitboard();
      case Piece::DRAGON:
        pos = static_cast<Square>(position.nonBitboardPieces.White.Dragon);
        return pos != Square::NONE ? Bitboard(pos) : Bitboard();
      default:
        return {0, 0, 0};
    }
  } else if (player == Player::Type::BLACK) {
    switch (piece) {
      case Piece::PAWN:
        return position.bbs[BB::Type::PAWN] & position.bbs[BB::Type::ALL_BLACK];
      case Piece::KNIGHT:
        return position.bbs[BB::Type::KNIGHT] &
               position.bbs[BB::Type::ALL_BLACK];
      case Piece::SILVER_GENERAL:
        return position.bbs[BB::Type::SILVER_GENERAL] &
               position.bbs[BB::Type::ALL_BLACK];
      case Piece::GOLD_GENERAL:
        return position.bbs[BB::Type::GOLD_GENERAL] &
               position.bbs[BB::Type::ALL_BLACK];
      case Piece::KING:
        pos = static_cast<Square>(position.nonBitboardPieces.Black.King);
        return pos != Square::NONE ? Bitboard(pos) : Bitboard();
      case Piece::LANCE:
        pos1 = static_cast<Square>(position.nonBitboardPieces.Black.Lance1);
        pos2 = static_cast<Square>(position.nonBitboardPieces.Black.Lance2);
        return (pos1 != Square::NONE ? Bitboard(pos1) : Bitboard()) |
               (pos2 != Square::NONE ? Bitboard(pos2) : Bitboard());
      case Piece::BISHOP:
        pos = static_cast<Square>(position.nonBitboardPieces.Black.Bishop);
        return pos != Square::NONE ? Bitboard(pos) : Bitboard();
      case Piece::ROOK:
        pos = static_cast<Square>(position.nonBitboardPieces.Black.Rook);
        return pos != Square::NONE ? Bitboard(pos) : Bitboard();
      case Piece::HORSE:
        pos = static_cast<Square>(position.nonBitboardPieces.Black.Horse);
        return pos != Square::NONE ? Bitboard(pos) : Bitboard();
      case Piece::DRAGON:
        pos = static_cast<Square>(position.nonBitboardPieces.Black.Dragon);
        return pos != Square::NONE ? Bitboard(pos) : Bitboard();
      default:
        return {0, 0, 0};
    }
  }
}
