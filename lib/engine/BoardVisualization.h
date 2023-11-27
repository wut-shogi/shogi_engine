#include "Board.h"

class BoardVisualizator {
 public:
  static void Show(const Board& position) {
    std::array<std::array<char, 3>, BOARD_SIZE> piecesStrings;
    for (int player = Player::Type::WHITE; player <= Player::Type::BLACK;
         player++) {
      for (int piece = Piece::Type::PAWN; piece <= Piece::Type::DRAGON;
           piece++) {
        Bitboard pieces = GetPiecesDynamic(static_cast<Piece::Type>(piece), static_cast<Player::Type>(player), position);
        BitboardIterator iterator(pieces);
        while (iterator.Next()) {
          if (iterator.IsCurrentSquareOccupied()) {
            piecesStrings[iterator.GetCurrentSquare()] ==
                pieceToText(static_cast<Piece::Type>(piece),
                              static_cast<Player::Type>(player));
          }
        }
      }
    }
  }

 private:
  static std::array<char, 3> pieceToText(Piece::Type piece,
                                         Player::Type player) {
    char basicSign;
    bool isPromoted;
    switch (piece) {
      case Piece::PAWN:
        basicSign = 'p';
        break;
      case Piece::KNIGHT:
        basicSign = 'n';
        break;
      case Piece::SILVER_GENERAL:
        basicSign = 's';
        break;
      case Piece::GOLD_GENERAL:
        basicSign = 'g';
        break;
      case Piece::KING:
        basicSign = 'k';
        break;
      case Piece::LANCE:
        basicSign = 'l';
        break;
      case Piece::BISHOP:
        basicSign = 'b';
        break;
      case Piece::ROOK:
        basicSign = 'r';
        break;
      case Piece::HORSE:
        isPromoted = true;
        basicSign = 'b';
        break;
      case Piece::DRAGON:
        isPromoted = true;
        basicSign = 'r';
        break;
      default:
        basicSign = 'e';
        break;
    }
  }
  };
