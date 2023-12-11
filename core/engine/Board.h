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
  Board() {}

  Board(std::array<Bitboard, BB::Type::SIZE>&& bbs, InHandLayout inHand)
      : inHand(inHand) {
    std::memcpy(this->bbs, bbs.data(), sizeof(this->bbs));
  }
  static Board FromSFEN(std::string SFENstring, bool& outIsWhite) {
    Board result;
    std::string boardString, player, captures;
    std::istringstream iss(SFENstring);
    iss >> boardString >> player >> captures;
    int rank = 0, file = 0;
    for (char c : boardString) {
      if (c == '/') {
        rank++;
        file = 0;
      } else if (isdigit(c)) {
        file += (c - '0');
      } else if (c == '+') {
        setSquare(result.bbs[BB::Type::PROMOTED], rankFileToSquare(rank, file));
      } else {
        if (c >= 'a' && c <= 'z') {
          setSquare(result.bbs[BB::Type::ALL_WHITE],
                    rankFileToSquare(rank, file));
        } else if (c >= 'A' && c <= 'Z') {
          setSquare(result.bbs[BB::Type::ALL_BLACK],
                    rankFileToSquare(rank, file));
          c += 32;
        }
        switch (c) {
          case 'p':
            setSquare(result.bbs[BB::Type::PAWN], rankFileToSquare(rank, file));
            break;
          case 'l':
            setSquare(result.bbs[BB::Type::LANCE],
                      rankFileToSquare(rank, file));
            break;
          case 'n':
            setSquare(result.bbs[BB::Type::KNIGHT],
                      rankFileToSquare(rank, file));
            break;
          case 's':
            setSquare(result.bbs[BB::Type::SILVER_GENERAL],
                      rankFileToSquare(rank, file));
            break;
          case 'g':
            setSquare(result.bbs[BB::Type::GOLD_GENERAL],
                      rankFileToSquare(rank, file));
            break;
          case 'b':
            setSquare(result.bbs[BB::Type::BISHOP],
                      rankFileToSquare(rank, file));
            break;
          case 'r':
            setSquare(result.bbs[BB::Type::ROOK], rankFileToSquare(rank, file));
            break;
          case 'k':
            setSquare(result.bbs[BB::Type::KING], rankFileToSquare(rank, file));
            break;
          default:
            break;
        }
        file++;
      }
    }

    int count = 1;
    for (char c : captures) {
      if (isdigit(c)) {
        count = c - '0';
      } else {
        switch (c) {
          case 'p':
            result.inHand.pieceNumber.WhitePawn = count;
            break;
          case 'l':
            result.inHand.pieceNumber.WhiteLance = count;
            break;
          case 'n':
            result.inHand.pieceNumber.WhiteKnight = count;
            break;
          case 's':
            result.inHand.pieceNumber.WhiteSilverGeneral = count;
            break;
          case 'g':
            result.inHand.pieceNumber.WhiteGoldGeneral = count;
            break;
          case 'b':
            result.inHand.pieceNumber.WhiteBishop = count;
            break;
          case 'r':
            result.inHand.pieceNumber.WhiteRook = count;
            break;
          case 'P':
            result.inHand.pieceNumber.BlackPawn = count;
            break;
          case 'L':
            result.inHand.pieceNumber.BlackLance = count;
            break;
          case 'N':
            result.inHand.pieceNumber.BlackKnight = count;
            break;
          case 'S':
            result.inHand.pieceNumber.BlackSilverGeneral = count;
            break;
          case 'G':
            result.inHand.pieceNumber.BlackGoldGeneral = count;
            break;
          case 'B':
            result.inHand.pieceNumber.BlackBishop = count;
            break;
          case 'R':
            result.inHand.pieceNumber.BlackRook = count;
            break;
          default:
            break;
        }
        count = 1;
      }
    }

    outIsWhite = player[0] == 'w' ? true : false;

    return result;
  }

  Bitboard& operator[](BB::Type idx) { return bbs[idx]; }

  const Bitboard& operator[](BB::Type idx) const { return bbs[idx]; }

  Board& operator=(const Board& board) {
    std::memcpy(this->bbs, board.bbs, sizeof(this->bbs));
    inHand.value = board.inHand.value;
    return *this;
  }
};

std::string boardToSFEN(const Board& board);
void print_Board(const Board& board);

namespace Boards {
Board STARTING_BOARD();
}
}  // namespace engine
}  // namespace shogi
