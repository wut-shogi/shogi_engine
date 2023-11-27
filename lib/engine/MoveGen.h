#pragma once
#include "Board.h"
#include "MoveGenHelpers.h"

struct Move {
  Piece::Type piece;
  Square from;
  Square to;
};

template<Player::Type player>
size_t GetNumberOfAllLegalMoves(const Board& position) {
  size_t numberOfMoves = 0;
  // Non-sliding pieces
  numberOfMoves +=
      GetNonSlidingPiecesNumberOfMoves<Piece::Type::PAWN, player>(position);
  std::cout << "Number of pawn moves: "
            << GetNonSlidingPiecesNumberOfMoves<Piece::Type::PAWN, player>(
                   position)
            << std::endl;
  numberOfMoves +=
      GetNonSlidingPiecesNumberOfMoves<Piece::Type::KNIGHT, player>(position);
  std::cout << "Number of knight moves: "
            << GetNonSlidingPiecesNumberOfMoves<Piece::Type::KNIGHT, player>(
                   position)
            << std::endl;
  numberOfMoves +=
      GetNonSlidingPiecesNumberOfMoves<Piece::Type::SILVER_GENERAL, player>(position);
  std::cout << "Number of silver general moves: "
      << GetNonSlidingPiecesNumberOfMoves<Piece::Type::SILVER_GENERAL, player>(
                   position)
            << std::endl;
  numberOfMoves +=
      GetNonSlidingPiecesNumberOfMoves<Piece::Type::GOLD_GENERAL, player>(position);
  std::cout << "Number of gold general moves: "
      << GetNonSlidingPiecesNumberOfMoves<Piece::Type::GOLD_GENERAL, player>(
                   position)
            << std::endl;
  numberOfMoves +=
      GetNonSlidingPiecesNumberOfMoves<Piece::Type::PROMOTED_PAWN, player>(position);
  std::cout << "Number of promoted pawn moves: "
      << GetNonSlidingPiecesNumberOfMoves<Piece::Type::PROMOTED_PAWN, player>(
                   position)
            << std::endl;
  numberOfMoves +=
      GetNonSlidingPiecesNumberOfMoves<Piece::Type::PROMOTED_LANCE, player>(
          position);
  std::cout << "Number of promoted lance moves: "
      << GetNonSlidingPiecesNumberOfMoves<Piece::Type::PROMOTED_LANCE, player>(
                   position)
            << std::endl;
  numberOfMoves +=
      GetNonSlidingPiecesNumberOfMoves<Piece::Type::PROMOTED_KNIGHT, player>(
          position);
  std::cout << "Number of promoted knight moves: "
      << GetNonSlidingPiecesNumberOfMoves<Piece::Type::PROMOTED_KNIGHT, player>(
                   position)
            << std::endl;
  numberOfMoves +=
      GetNonSlidingPiecesNumberOfMoves<Piece::Type::PROMOTED_SILVER_GENERAL, player>(
          position);
  std::cout << "Number of promoted silver general moves: "
            << GetNonSlidingPiecesNumberOfMoves<
                   Piece::Type::PROMOTED_SILVER_GENERAL, player>(
                   position)
            << std::endl;

  // Sliding pieces
  numberOfMoves +=
      GetSlidingPiecesNumberOfMoves<Piece::Type::LANCE, player>(position);
  std::cout << "Number of lance moves: "
            << GetSlidingPiecesNumberOfMoves<Piece::Type::LANCE, player>(
                   position)
            << std::endl;
  numberOfMoves +=
      GetSlidingPiecesNumberOfMoves<Piece::Type::BISHOP, player>(position);
  std::cout << "Number of bishop moves: "
            << GetSlidingPiecesNumberOfMoves<Piece::Type::BISHOP, player>(
                   position)
            << std::endl;
  numberOfMoves +=
      GetSlidingPiecesNumberOfMoves<Piece::Type::ROOK, player>(
          position);
  std::cout << "Number of rook moves: "
            << GetSlidingPiecesNumberOfMoves<Piece::Type::ROOK, player>(
                   position)
            << std::endl;
  numberOfMoves +=
      GetSlidingPiecesNumberOfMoves<Piece::Type::HORSE, player>(
          position);
  std::cout << "Number of horse moves: "
            << GetSlidingPiecesNumberOfMoves<Piece::Type::HORSE, player>(
                   position)
            << std::endl;
  numberOfMoves +=
      GetSlidingPiecesNumberOfMoves<Piece::Type::DRAGON, player>(
          position);
  std::cout << "Number of dragon moves: "
            << GetSlidingPiecesNumberOfMoves<Piece::Type::DRAGON, player>(
                   position)
            << std::endl;

  // Drop moves
  numberOfMoves += GetDropPiecesNumberOfMoves<player>(position);
  std::cout << "Number of drop moves: "
            << GetDropPiecesNumberOfMoves<player>(position)
            << std::endl;

  return numberOfMoves;
}


size_t countAllMoves(const Board& board, bool isWhite);



