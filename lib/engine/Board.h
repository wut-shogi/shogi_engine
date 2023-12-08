#pragma once
#include <cassert>
#include <sstream>
#include "Bitboard.h"
#include "Rules.h"

struct Board {
  Bitboard bbs[BB::Type::SIZE];
  InHandPieces inHandPieces;
  Board() {}

  Board(std::array<Bitboard, BB::Type::SIZE>&& bbs, InHandPieces inHandPieces)
      : inHandPieces(inHandPieces) {
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
    PlayerInHandPieces* playerInHandPieces;
    for (char c : captures) {
      if (isdigit(c)) {
        count = c - '0';
      } else {
        if (c >= 'A' && c <= 'Z') {
          playerInHandPieces = &result.inHandPieces.Black;
          c += 32;
        } else {
          playerInHandPieces = &result.inHandPieces.White;
        }
        switch (c) {
          case 'p':
            playerInHandPieces->Pawn = count;
            break;
          case 'l':
            playerInHandPieces->Lance = count;
            break;
          case 'n':
            playerInHandPieces->Knight = count;
            break;
          case 's':
            playerInHandPieces->SilverGeneral = count;
            break;
          case 'g':
            playerInHandPieces->GoldGeneral = count;
            break;
          case 'b':
            playerInHandPieces->Bishop = count;
            break;
          case 'r':
            playerInHandPieces->Rook = count;
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
    return *this;
  }
};

std::string boardToSFEN(const Board& board);
void print_Board(const Board& board);

namespace Boards {
Board STARTING_BOARD();
}
