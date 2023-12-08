#include "MoveGen.h"

size_t countWhiteMoves(const Board& board,
                       Bitboard& outValidMoves,
                       Bitboard& attackedByEnemy) {
  Bitboard king = board[BB::Type::KING] & board[BB::Type::ALL_WHITE];
  Bitboard notPromoted = ~board[BB::Type::PROMOTED];
  Bitboard checkingPieces, attacked, pieces, moves, slidingChecksPaths, attacks,
      attacksFull, mask, potentialPin, pinned, ourAttacks;
  BitboardIterator iterator;
  Square square;
  size_t numberOfMoves = 0;

  // Non Sliding pieces
  // Pawns
  pieces =
      board.bbs[BB::Type::PAWN] & board.bbs[BB::Type::ALL_BLACK] & notPromoted;
  checkingPieces |= moveS(king) & pieces;
  attacked |= moveN(pieces);
  // Knights
  pieces = board.bbs[BB::Type::KNIGHT] & board.bbs[BB::Type::ALL_BLACK] &
           notPromoted;
  checkingPieces |= moveS(moveSE(king) | moveSW(king)) & pieces;
  attacked |= moveN(moveNE(pieces) | moveNW(pieces));
  // Silve generals
  pieces = board.bbs[BB::Type::SILVER_GENERAL] &
           board.bbs[BB::Type::ALL_BLACK] & notPromoted;
  checkingPieces |= (moveSE(king) | moveS(king) | moveSW(king) | moveNE(king) |
                     moveNW(king)) &
                    pieces;
  attacked |= moveNE(pieces) | moveN(pieces) | moveNW(pieces) | moveSE(pieces) |
              moveSW(pieces);
  // Gold generals
  pieces = (board[BB::Type::GOLD_GENERAL] |
            ((board[BB::Type::PAWN] | board[BB::Type::KNIGHT] |
              board[BB::Type::SILVER_GENERAL] | board[BB::Type::LANCE]) &
             board[BB::Type::PROMOTED])) &
           board.bbs[BB::Type::ALL_BLACK];
  checkingPieces |= (moveSE(king) | moveS(king) | moveSW(king) | moveE(king) |
                     moveW(king) | moveN(king)) &
                    pieces;
  attacked |= moveNE(pieces) | moveN(pieces) | moveNW(pieces) | moveE(pieces) |
              moveW(pieces) | moveS(pieces);
  // Horse (non sliding part)
  pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
           board[BB::Type::PROMOTED];
  checkingPieces |=
      (moveN(king) | moveE(king) | moveS(king) | moveW(king)) & pieces;
  attacked |= moveN(pieces) | moveE(pieces) | moveS(pieces) | moveW(pieces);
  // Dragon (non sliding part)
  pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
           board[BB::Type::PROMOTED];
  checkingPieces |=
      (moveNW(king) | moveNE(king) | moveSE(king) | moveSW(king)) & pieces;
  attacked |= moveNW(pieces) | moveNE(pieces) | moveSE(pieces) | moveSW(pieces);

  // Sliding pieces
  iterator.Init(king);
  iterator.Next();
  Square kingSquare = iterator.GetCurrentSquare();
  Bitboard occupied = board[BB::Type::ALL_BLACK] | board[BB::Type::ALL_WHITE];
  // Lance
  {
    pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] & notPromoted;
    checkingPieces |= getFileAttacks(kingSquare, occupied) &
                      ~getRankMask(squareToRank(kingSquare)) & pieces;
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      attacksFull = getFileAttacks(square, occupied);
      attacked |= attacksFull;
      mask = getRankMask(squareToRank(square));
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::ALL_WHITE];
        attacks = getFileAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
    }
  }

  // Rook and dragon
  {
    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK];
    checkingPieces |= (getRankAttacks(kingSquare, occupied) |
                       getFileAttacks(kingSquare, occupied)) &
                      pieces;
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      // Check if king is in check without white pieces
      // We have to check all 4 directions
      // left-right
      attacksFull = getRankAttacks(square, occupied);
      attacked |= attacksFull;
      mask = getFileMask(squareToFile(square));
      // left
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & mask;
        attacks = getRankAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
      // right
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & ~mask;
        attacks = getRankAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & ~mask;
      }
      // up-down
      attacksFull = getFileAttacks(square, occupied);
      attacked |= attacksFull;
      mask = getRankMask(squareToRank(square));
      // up
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & mask;
        attacks = getFileAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
      // down
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & ~mask;
        attacks = getFileAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & ~mask;
      }
    }
  }

  // Bishop and horse pins
  {
    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK];
    checkingPieces |= (getDiagRightAttacks(kingSquare, occupied) |
                       getDiagLeftAttacks(kingSquare, occupied)) &
                      pieces;
    iterator.Init(board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK]);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      // Check if king is in check without white pieces
      // We have to check all 4 directions
      // right diag
      attacksFull = getDiagRightAttacks(square, occupied);
      attacked |= attacksFull;
      mask = ~getFileMask(squareToFile(square)) &
             getRankMask(squareToRank(square));
      // SW
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & mask;
        attacks = getDiagRightAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
      // NE
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & ~mask;
        attacks = getDiagRightAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & ~mask;
      }
      // left diag
      attacksFull = getDiagLeftAttacks(square, occupied);
      attacked |= attacksFull;
      mask =
          getFileMask(squareToFile(square)) & getRankMask(squareToRank(square));
      // NW
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & mask;
        attacks = getDiagLeftAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
      // SE
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & ~mask;
        attacks = getDiagLeftAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & ~mask;
      }
    }
  }

  int numberOfCheckingPieces = std::popcount<uint32_t>(checkingPieces[TOP]) +
                               std::popcount<uint32_t>(checkingPieces[MID]) +
                               std::popcount<uint32_t>(checkingPieces[BOTTOM]);

  // King can always move to non attacked squares
  moves = moveNE(king) | moveN(king) | moveNW(king) | moveE(king) |
          moveW(king) | moveSE(king) | moveS(king) | moveSW(king);
  moves &= ~attacked & ~board[BB::Type::ALL_WHITE];
  numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
                   std::popcount<uint32_t>(moves[MID]) +
                   std::popcount<uint32_t>(moves[BOTTOM]);
  std::cout << "After king: " << numberOfMoves << std::endl;
  Bitboard validMoves;
  if (numberOfCheckingPieces == 1) {
    // if king is checked by exactly one piece legal moves can also be block
    // sliding check or capture a checking piece
    validMoves = checkingPieces | (slidingChecksPaths & ~king);
  } else if (numberOfCheckingPieces == 0) {
    // If there is no checks all moves are valid (you cannot capture your own
    // piece)
    validMoves = ~board[BB::Type::ALL_WHITE];
  }

  outValidMoves = validMoves;
  attackedByEnemy = attacked;

  // Pawn moves
  {
    pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] & notPromoted;
    moves = moveS(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
                     std::popcount<uint32_t>(moves[MID]) +
                     std::popcount<uint32_t>(moves[BOTTOM] & BOTTOM_RANK) *
                         2 +  // promotions
                     std::popcount<uint32_t>(moves[BOTTOM] &
                                             ~BOTTOM_RANK);  // forced promotion
  }
  std::cout << "After pawns: " << numberOfMoves << std::endl;

  // Knight moves
  {
    pieces = board[BB::Type::KNIGHT] & board[BB::Type::ALL_WHITE] & notPromoted;
    moves = moveS(moveSE(pieces)) & validMoves;
    ourAttacks |= moves;
    numberOfMoves +=
        std::popcount<uint32_t>(moves[TOP]) +
        std::popcount<uint32_t>(moves[MID]) +
        std::popcount<uint32_t>(moves[BOTTOM] & TOP_RANK) * 2 +  // promotions
        std::popcount<uint32_t>(moves[BOTTOM] & ~TOP_RANK);  // forced promotion
    moves = moveS(moveSW(pieces)) & validMoves;
    ourAttacks |= moves;
    numberOfMoves +=
        std::popcount<uint32_t>(moves[TOP]) +
        std::popcount<uint32_t>(moves[MID]) +
        std::popcount<uint32_t>(moves[BOTTOM] & TOP_RANK) * 2 +  // promotions
        std::popcount<uint32_t>(moves[BOTTOM] & ~TOP_RANK);  // forced promotion
  }
  std::cout << "After knights: " << numberOfMoves << std::endl;

  // SilverGenerals moves
  {
    pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_WHITE] &
             notPromoted;
    moves = moveS(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
                     std::popcount<uint32_t>(moves[MID]) +
                     std::popcount<uint32_t>(moves[BOTTOM]) * 2;  // promotions
    moves = moveSE(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
                     std::popcount<uint32_t>(moves[MID]) +
                     std::popcount<uint32_t>(moves[BOTTOM]) * 2;  // promotions
    moves = moveSW(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
                     std::popcount<uint32_t>(moves[MID]) +
                     std::popcount<uint32_t>(moves[BOTTOM]) * 2;  // promotions
    moves = moveNE(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
                     std::popcount<uint32_t>(moves[MID] & BOTTOM_RANK) *
                         2 +  // promotion when starting from promotion zone
                     std::popcount<uint32_t>(moves[MID] & ~BOTTOM_RANK) +
                     std::popcount<uint32_t>(moves[BOTTOM]) * 2;  // promotions
    moves = moveNW(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
                     std::popcount<uint32_t>(moves[MID] & BOTTOM_RANK) *
                         2 +  // promotion when starting from promotion zone
                     std::popcount<uint32_t>(moves[MID] & ~BOTTOM_RANK) +
                     std::popcount<uint32_t>(moves[BOTTOM]) * 2;  // promotions
  }
  std::cout << "After silverGens: " << numberOfMoves << std::endl;

  // GoldGenerals moves
  {
    pieces = (board[BB::Type::GOLD_GENERAL] |
              ((board[BB::Type::PAWN] | board[BB::Type::LANCE] |
                board[BB::Type::KNIGHT] | board[BB::Type::SILVER_GENERAL]) &
               board[BB::Type::PROMOTED])) &
             board[BB::Type::ALL_WHITE];
    moves = moveS(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
                     std::popcount<uint32_t>(moves[MID]) +
                     std::popcount<uint32_t>(moves[BOTTOM]);
    moves = moveSE(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
                     std::popcount<uint32_t>(moves[MID]) +
                     std::popcount<uint32_t>(moves[BOTTOM]);
    moves = moveSW(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
                     std::popcount<uint32_t>(moves[MID]) +
                     std::popcount<uint32_t>(moves[BOTTOM]);
    moves = moveE(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
                     std::popcount<uint32_t>(moves[MID]) +
                     std::popcount<uint32_t>(moves[BOTTOM]);
    moves = moveW(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
                     std::popcount<uint32_t>(moves[MID]) +
                     std::popcount<uint32_t>(moves[BOTTOM]);
    moves = moveN(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
                     std::popcount<uint32_t>(moves[MID]) +
                     std::popcount<uint32_t>(moves[BOTTOM]);
  }
  std::cout << "After goldGens: " << numberOfMoves << std::endl;

  // Lance moves
  {
    pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] & notPromoted;
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      moves = getFileAttacks(square, occupied) &
              ~getRankMask(squareToRank(square)) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
                       std::popcount<uint32_t>(moves[MID]) +
                       std::popcount<uint32_t>(moves[BOTTOM] & BOTTOM_RANK) *
                           2 +  // promotions
                       std::popcount<uint32_t>(
                           moves[BOTTOM] & ~BOTTOM_RANK);  // forced promotion
    }
  }
  std::cout << "After lances: " << numberOfMoves << std::endl;

  // Bishop moves
  {
    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] & notPromoted;
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      moves = (getDiagRightAttacks(square, occupied) |
               getDiagLeftAttacks(square, occupied)) &
              validMoves;
      ourAttacks |= moves;
      if (square > WHITE_PROMOTION_START) {  // Starting from promotion zone
        numberOfMoves += (std::popcount<uint32_t>(moves[TOP]) +
                          std::popcount<uint32_t>(moves[MID]) +
                          std::popcount<uint32_t>(moves[BOTTOM])) *
                         2;
      } else {
        numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
                         std::popcount<uint32_t>(moves[MID]) +
                         std::popcount<uint32_t>(moves[BOTTOM]) *
                             2;  // end in promotion Zone
      }
    }
  }
  std::cout << "After bishop: " << numberOfMoves << std::endl;

  // Rook moves
  {
    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] & notPromoted;
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      moves = (getRankAttacks(square, occupied) |
               getFileAttacks(square, occupied)) &
              validMoves;
      ourAttacks |= moves;
      if (square > WHITE_PROMOTION_START) {  // Starting from promotion zone
        numberOfMoves += (std::popcount<uint32_t>(moves[TOP]) +
                          std::popcount<uint32_t>(moves[MID]) +
                          std::popcount<uint32_t>(moves[BOTTOM])) *
                         2;
      } else {
        numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
                         std::popcount<uint32_t>(moves[MID]) +
                         std::popcount<uint32_t>(moves[BOTTOM]) *
                             2;  // end in promotion Zone
      }
    }
  }
  std::cout << "After rook: " << numberOfMoves << std::endl;

  // Horse moves
  {
    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
             board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      Bitboard horse = Bitboard(square);
      moves = (getDiagRightAttacks(square, occupied) |
               getDiagLeftAttacks(square, occupied) | moveN(horse) |
               moveE(horse) | moveS(horse) | moveW(horse)) &
              validMoves;
      ourAttacks |= moves;
      numberOfMoves += (std::popcount<uint32_t>(moves[TOP]) +
                        std::popcount<uint32_t>(moves[MID]) +
                        std::popcount<uint32_t>(moves[BOTTOM]));
    }
  }
  std::cout << "After horse: " << numberOfMoves << std::endl;

  // Dragon moves
  {
    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
             board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      Bitboard dragon(square);
      moves =
          (getRankAttacks(square, occupied) | getFileAttacks(square, occupied) |
           moveNW(dragon) | moveNE(dragon) | moveSE(dragon) | moveSW(dragon)) &
          validMoves;
      ourAttacks |= moves;
      numberOfMoves += (std::popcount<uint32_t>(moves[TOP]) +
                        std::popcount<uint32_t>(moves[MID]) +
                        std::popcount<uint32_t>(moves[BOTTOM]));
    }
  }
  std::cout << "After dragon: " << numberOfMoves << std::endl;

  // Drop moves
  {
    Bitboard legalDropSpots;
    // Pawns
    if (board.inHandPieces.White.Pawn > 0) {
      legalDropSpots = validMoves;
      // Cannot drop on last rank
      legalDropSpots[BOTTOM] &= ~BOTTOM_RANK;
      // Cannot drop to give checkmate
      pieces = board[BB::Type::KING] & board[BB::Type::ALL_BLACK];
      // All valid enemy king moves
      moves =
          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
          ourAttacks;
      // If there is only one spot pawn cannot block it
      if (std::popcount<uint32_t>(moves[TOP]) +
              std::popcount<uint32_t>(moves[MID]) +
              std::popcount<uint32_t>(moves[BOTTOM]) ==
          1) {
        legalDropSpots &= ~moveN(moves);
      }
      // Cannot drop on file with other pawn
      Bitboard validFiles;
      for (int fileIdx = 0; fileIdx < BOARD_DIM; fileIdx++) {
        Bitboard file =
            getFileAttacks(static_cast<Square>(fileIdx), validFiles);
        if (file & board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
            notPromoted) {
          validFiles |= file;
        }
      }
      legalDropSpots &= validFiles;
      numberOfMoves += std::popcount<uint32_t>(legalDropSpots[TOP]) +
                       std::popcount<uint32_t>(legalDropSpots[MID]) +
                       std::popcount<uint32_t>(legalDropSpots[BOTTOM]);
    }
    if (board.inHandPieces.White.Lance > 0) {
      legalDropSpots = validMoves;
      // Cannot drop on last rank
      legalDropSpots[BOTTOM] &= ~BOTTOM_RANK;
      numberOfMoves += std::popcount<uint32_t>(legalDropSpots[TOP]) +
                       std::popcount<uint32_t>(legalDropSpots[MID]) +
                       std::popcount<uint32_t>(legalDropSpots[BOTTOM]);
    }
    if (board.inHandPieces.White.Knight > 0) {
      legalDropSpots = validMoves;
      // Cannot drop on last two ranks
      legalDropSpots[BOTTOM] &= TOP_RANK;
      numberOfMoves += std::popcount<uint32_t>(legalDropSpots[TOP]) +
                       std::popcount<uint32_t>(legalDropSpots[MID]) +
                       std::popcount<uint32_t>(legalDropSpots[BOTTOM]);
    }
    legalDropSpots = validMoves;
    numberOfMoves += ((board.inHandPieces.White.SilverGeneral > 0) +
                      (board.inHandPieces.White.GoldGeneral > 0) +
                      (board.inHandPieces.White.Bishop > 0) +
                      (board.inHandPieces.White.Rook > 0)) *
                     (std::popcount<uint32_t>(legalDropSpots[TOP]) +
                      std::popcount<uint32_t>(legalDropSpots[MID]) +
                      std::popcount<uint32_t>(legalDropSpots[BOTTOM]));
  }

  return numberOfMoves;
}

size_t countBlackMoves(const Board& board,
                       Bitboard& outValidMoves,
                       Bitboard& attackedByEnemy) {
  Bitboard king = board[BB::Type::KING] & board[BB::Type::ALL_BLACK];
  Bitboard notPromoted = ~board[BB::Type::PROMOTED];
  Bitboard checkingPieces, attacked, pieces, moves, slidingChecksPaths, attacks,
      attacksFull, mask, potentialPin, pinned, ourAttacks;
  BitboardIterator iterator;
  Square square;
  size_t numberOfMoves = 0;

  // Non Sliding pieces
  // Pawns
  pieces =
      board.bbs[BB::Type::PAWN] & board.bbs[BB::Type::ALL_WHITE] & notPromoted;
  checkingPieces |= moveN(king) & pieces;
  attacked |= moveS(pieces);
  // Knights
  pieces = board.bbs[BB::Type::KNIGHT] & board.bbs[BB::Type::ALL_WHITE] &
           notPromoted;
  checkingPieces |= moveN(moveNE(king) | moveNW(king)) & pieces;
  attacked |= moveS(moveSE(pieces) | moveSW(pieces));
  // Silver generals
  pieces = board.bbs[BB::Type::SILVER_GENERAL] &
           board.bbs[BB::Type::ALL_WHITE] & notPromoted;
  checkingPieces |= (moveNE(king) | moveN(king) | moveNW(king) | moveSE(king) |
                     moveSW(king)) &
                    pieces;
  attacked |= moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveNE(pieces) |
              moveNW(pieces);
  // gold generals
  pieces = (board[BB::Type::GOLD_GENERAL] |
            ((board[BB::Type::PAWN] | board[BB::Type::KNIGHT] |
              board[BB::Type::SILVER_GENERAL] | board[BB::Type::LANCE]) &
             board[BB::Type::PROMOTED])) &
           board.bbs[BB::Type::ALL_WHITE];
  checkingPieces |= (moveNE(king) | moveN(king) | moveNW(king) | moveE(king) |
                     moveW(king) | moveS(king)) &
                    pieces;
  attacked |= moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveE(pieces) |
              moveW(pieces) | moveN(pieces);
  // Horse (non sliding part)
  pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
           board[BB::Type::PROMOTED];
  checkingPieces |=
      (moveN(king) | moveE(king) | moveS(king) | moveW(king)) & pieces;
  attacked |= moveN(pieces) | moveE(pieces) | moveS(pieces) | moveW(pieces);
  // Dragon (non sldiing part)
  pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
           board[BB::Type::PROMOTED];
  checkingPieces |=
      (moveNW(king) | moveNE(king) | moveSE(king) | moveSW(king)) & pieces;
  attacked |= moveNW(pieces) | moveNE(pieces) | moveSE(pieces) | moveSW(pieces);

  // Sliding pieces
  iterator.Init(king);
  iterator.Next();
  Square kingSquare = iterator.GetCurrentSquare();
  Bitboard occupied = board[BB::Type::ALL_BLACK] | board[BB::Type::ALL_WHITE];
  // Lance
  {
    pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] & notPromoted;
    checkingPieces |= getFileAttacks(kingSquare, occupied) &
                      getRankMask(squareToRank(kingSquare)) & pieces;
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      attacksFull = getFileAttacks(square, occupied);
      attacked |= attacksFull;
      mask = ~getRankMask(squareToRank(square));
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::ALL_BLACK];
        attacks = getFileAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
    }
  }

  // Rook and dragon
  {
    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE];
    checkingPieces |= (getRankAttacks(kingSquare, occupied) |
                       getFileAttacks(kingSquare, occupied)) &
                      pieces;
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      // Check if king is in check without white pieces
      // We have to check all 4 directions
      // left-right
      attacksFull = getRankAttacks(square, occupied);
      attacked |= attacksFull;
      mask = getFileMask(squareToFile(square));
      // left
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & mask;
        attacks = getRankAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
      // right
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & ~mask;
        attacks = getRankAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & ~mask;
      }
      // up-down
      attacksFull = getFileAttacks(square, occupied);
      attacked |= attacksFull;
      mask = getRankMask(squareToRank(square));
      // up
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & mask;
        attacks = getFileAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
      // down
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & ~mask;
        attacks = getFileAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & ~mask;
      }
    }
  }

  // Bishop and horse pins
  {
    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE];
    checkingPieces |= (getDiagRightAttacks(kingSquare, occupied) |
                       getDiagLeftAttacks(kingSquare, occupied)) &
                      pieces;
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      // Check if king is in check without white pieces
      // We have to check all 4 directions
      // right diag
      attacksFull = getDiagRightAttacks(square, occupied);
      attacked |= attacksFull;
      mask = ~getFileMask(squareToFile(square)) &
             getRankMask(squareToRank(square));
      // SW
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & mask;
        attacks = getDiagRightAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
      // NE
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & ~mask;
        attacks = getDiagRightAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & ~mask;
      }
      // left diag
      attacksFull = getDiagLeftAttacks(square, occupied);
      attacked |= attacksFull;
      mask =
          getFileMask(squareToFile(square)) & getRankMask(squareToRank(square));
      // NW
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & mask;
        attacks = getDiagLeftAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
      // SE
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & ~mask;
        attacks = getDiagLeftAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & ~mask;
      }
    }
  }

  int numberOfCheckingPieces = std::popcount<uint32_t>(checkingPieces[TOP]) +
                               std::popcount<uint32_t>(checkingPieces[MID]) +
                               std::popcount<uint32_t>(checkingPieces[BOTTOM]);

  // King can always move to non attacked squares
  moves = moveNE(king) | moveN(king) | moveNW(king) | moveE(king) |
          moveW(king) | moveSE(king) | moveS(king) | moveSW(king);
  moves &= ~attacked & ~board[BB::Type::ALL_BLACK];
  numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
                   std::popcount<uint32_t>(moves[MID]) +
                   std::popcount<uint32_t>(moves[BOTTOM]);
  std::cout << "After king: " << numberOfMoves << std::endl;
  Bitboard validMoves;
  if (numberOfCheckingPieces == 1) {
    // if king is checked by exactly one piece legal moves can also be block
    // sliding check or capture a checking piece
    validMoves = checkingPieces | (slidingChecksPaths & ~king);
  } else if (numberOfCheckingPieces == 0) {
    // If there is no checks all moves are valid (you cannot capture your own
    // piece)
    validMoves = ~board[BB::Type::ALL_BLACK];
  }

  outValidMoves = validMoves;
  attackedByEnemy = attacked;

  // Pawn moves
  {
    pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] & notPromoted;
    moves = moveN(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves +=
        std::popcount<uint32_t>(moves[TOP] & TOP_RANK) +  // forced promotions
        std::popcount<uint32_t>(moves[TOP] & ~TOP_RANK) * 2 +  // promotions
        std::popcount<uint32_t>(moves[MID]) +
        std::popcount<uint32_t>(moves[BOTTOM]);
  }
  std::cout << "After pawns: " << numberOfMoves << std::endl;

  // Knight moves
  {
    pieces = board[BB::Type::KNIGHT] & board[BB::Type::ALL_BLACK] & notPromoted;
    moves = moveN(moveNE(pieces)) & validMoves;
    ourAttacks |= moves;
    numberOfMoves +=
        std::popcount<uint32_t>(moves[TOP] & TOP_RANK) +  // forced promotions
        std::popcount<uint32_t>(moves[TOP] & ~TOP_RANK) * 2 +  // promotions
        std::popcount<uint32_t>(moves[MID]) +
        std::popcount<uint32_t>(moves[BOTTOM]);
    moves = moveN(moveNW(pieces)) & validMoves;
    ourAttacks |= moves;
    numberOfMoves +=
        std::popcount<uint32_t>(moves[TOP] & TOP_RANK) +  // forced promotions
        std::popcount<uint32_t>(moves[TOP] & ~TOP_RANK) * 2 +  // promotions
        std::popcount<uint32_t>(moves[MID]) +
        std::popcount<uint32_t>(moves[BOTTOM]);
  }
  std::cout << "After knights: " << numberOfMoves << std::endl;

  // SilverGenerals moves
  {
    pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_BLACK] &
             notPromoted;
    moves = moveN(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) * 2 +  // promotions
                     std::popcount<uint32_t>(moves[MID]) +
                     std::popcount<uint32_t>(moves[BOTTOM]);
    moves = moveNE(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) * 2 +  // promotions
                     std::popcount<uint32_t>(moves[MID]) +
                     std::popcount<uint32_t>(moves[BOTTOM]);
    moves = moveNW(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) * 2 +  // promotions
                     std::popcount<uint32_t>(moves[MID]) +
                     std::popcount<uint32_t>(moves[BOTTOM]);
    moves = moveSE(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) * 2 +  // promotions
                     std::popcount<uint32_t>(moves[MID] & TOP_RANK) *
                         2 +  // promotion when starting from promotion zone
                     std::popcount<uint32_t>(moves[MID] & ~TOP_RANK) +
                     std::popcount<uint32_t>(moves[BOTTOM]);
    moves = moveSW(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) * 2 +  // promotions
                     std::popcount<uint32_t>(moves[MID] & TOP_RANK) *
                         2 +  // promotion when starting from promotion zone
                     std::popcount<uint32_t>(moves[MID] & ~TOP_RANK) +
                     std::popcount<uint32_t>(moves[BOTTOM]);
  }
  std::cout << "After silverGens: " << numberOfMoves << std::endl;

  // GoldGenerals moves
  {
    pieces = (board[BB::Type::GOLD_GENERAL] |
              ((board[BB::Type::PAWN] | board[BB::Type::LANCE] |
                board[BB::Type::KNIGHT] | board[BB::Type::SILVER_GENERAL]) &
               board[BB::Type::PROMOTED])) &
             board[BB::Type::ALL_BLACK];
    moves = moveN(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
                     std::popcount<uint32_t>(moves[MID]) +
                     std::popcount<uint32_t>(moves[BOTTOM]);
    moves = moveNE(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
                     std::popcount<uint32_t>(moves[MID]) +
                     std::popcount<uint32_t>(moves[BOTTOM]);
    moves = moveNW(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
                     std::popcount<uint32_t>(moves[MID]) +
                     std::popcount<uint32_t>(moves[BOTTOM]);
    moves = moveE(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
                     std::popcount<uint32_t>(moves[MID]) +
                     std::popcount<uint32_t>(moves[BOTTOM]);
    moves = moveW(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
                     std::popcount<uint32_t>(moves[MID]) +
                     std::popcount<uint32_t>(moves[BOTTOM]);
    moves = moveS(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
                     std::popcount<uint32_t>(moves[MID]) +
                     std::popcount<uint32_t>(moves[BOTTOM]);
  }
  std::cout << "After goldGens: " << numberOfMoves << std::endl;

  // Lance moves
  {
    pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] & notPromoted;
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      moves = getFileAttacks(square, occupied) &
              getRankMask(squareToRank(square)) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          std::popcount<uint32_t>(moves[TOP] & TOP_RANK) +  // forced promotions
          std::popcount<uint32_t>(moves[TOP] & ~TOP_RANK) * 2 +  // promotions
          std::popcount<uint32_t>(moves[MID]) +
          std::popcount<uint32_t>(moves[BOTTOM]);
    }
  }
  std::cout << "After lances: " << numberOfMoves << std::endl;

  // Bishop moves
  {
    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] & notPromoted;
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      moves = (getDiagRightAttacks(square, occupied) |
               getDiagLeftAttacks(square, occupied)) &
              validMoves;
      ourAttacks |= moves;
      if (square < BLACK_PROMOTION_END) {  // Starting from promotion zone
        numberOfMoves += (std::popcount<uint32_t>(moves[TOP]) +
                          std::popcount<uint32_t>(moves[MID]) +
                          std::popcount<uint32_t>(moves[BOTTOM])) *
                         2;
      } else {
        numberOfMoves +=
            std::popcount<uint32_t>(moves[TOP]) * 2 +  // end in promotion Zone
            std::popcount<uint32_t>(moves[MID]) +
            std::popcount<uint32_t>(moves[BOTTOM]);
      }
    }
  }
  std::cout << "After bishop: " << numberOfMoves << std::endl;

  // Rook moves
  {
    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] & notPromoted;
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      moves = (getRankAttacks(square, occupied) |
               getFileAttacks(square, occupied)) &
              validMoves;
      ourAttacks |= moves;
      if (square < BLACK_PROMOTION_END) {  // Starting from promotion zone
        numberOfMoves += (std::popcount<uint32_t>(moves[TOP]) +
                          std::popcount<uint32_t>(moves[MID]) +
                          std::popcount<uint32_t>(moves[BOTTOM])) *
                         2;
      } else {
        numberOfMoves +=
            std::popcount<uint32_t>(moves[TOP]) * 2 +  // end in promotion Zone
            std::popcount<uint32_t>(moves[MID]) +
            std::popcount<uint32_t>(moves[BOTTOM]);
      }
    }
  }
  std::cout << "After rook: " << numberOfMoves << std::endl;

  // Horse moves
  {
    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
             board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      Bitboard horse = Bitboard(square);
      moves = (getDiagRightAttacks(square, occupied) |
               getDiagLeftAttacks(square, occupied) | moveN(horse) |
               moveE(horse) | moveS(horse) | moveW(horse)) &
              validMoves;
      ourAttacks |= moves;
      numberOfMoves += (std::popcount<uint32_t>(moves[TOP]) +
                        std::popcount<uint32_t>(moves[MID]) +
                        std::popcount<uint32_t>(moves[BOTTOM]));
    }
  }
  std::cout << "After horse: " << numberOfMoves << std::endl;

  // Dragon moves
  {
    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
             board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      Bitboard dragon(square);
      moves =
          (getRankAttacks(square, occupied) | getFileAttacks(square, occupied) |
           moveNW(dragon) | moveNE(dragon) | moveSE(dragon) | moveSW(dragon)) &
          validMoves;
      ourAttacks |= moves;
      numberOfMoves += (std::popcount<uint32_t>(moves[TOP]) +
                        std::popcount<uint32_t>(moves[MID]) +
                        std::popcount<uint32_t>(moves[BOTTOM]));
    }
  }
  std::cout << "After dragon: " << numberOfMoves << std::endl;
  // Drop moves
  {
    Bitboard legalDropSpots;
    // Pawns
    if (board.inHandPieces.Black.Pawn > 0) {
      legalDropSpots = validMoves;
      // Cannot drop on last rank
      legalDropSpots[TOP] &= ~TOP_RANK;
      // Cannot drop to give checkmate
      pieces = board[BB::Type::KING] & board[BB::Type::ALL_WHITE];
      // All valid enemy king moves
      moves =
          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
          ourAttacks;
      // If there is only one spot pawn cannot block it
      if (std::popcount<uint32_t>(moves[TOP]) +
              std::popcount<uint32_t>(moves[MID]) +
              std::popcount<uint32_t>(moves[BOTTOM]) ==
          1) {
        legalDropSpots &= ~moveS(moves);
      }
      // Cannot drop on file with other pawn
      Bitboard validFiles;
      for (int fileIdx = 0; fileIdx < BOARD_DIM; fileIdx++) {
        Bitboard file =
            getFileAttacks(static_cast<Square>(fileIdx), validFiles);
        if (file & board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
            notPromoted) {
          validFiles |= file;
        }
      }
      legalDropSpots &= validFiles;
      numberOfMoves += std::popcount<uint32_t>(legalDropSpots[TOP]) +
                       std::popcount<uint32_t>(legalDropSpots[MID]) +
                       std::popcount<uint32_t>(legalDropSpots[BOTTOM]);
    }
    if (board.inHandPieces.Black.Lance > 0) {
      legalDropSpots = validMoves;
      // Cannot drop on last rank
      legalDropSpots[TOP] &= ~TOP_RANK;
      numberOfMoves += std::popcount<uint32_t>(legalDropSpots[TOP]) +
                       std::popcount<uint32_t>(legalDropSpots[MID]) +
                       std::popcount<uint32_t>(legalDropSpots[BOTTOM]);
    }
    if (board.inHandPieces.Black.Knight > 0) {
      legalDropSpots = validMoves;
      // Cannot drop on last two ranks
      legalDropSpots[TOP] &= TOP_RANK;
      numberOfMoves += std::popcount<uint32_t>(legalDropSpots[TOP]) +
                       std::popcount<uint32_t>(legalDropSpots[MID]) +
                       std::popcount<uint32_t>(legalDropSpots[BOTTOM]);
    }
    legalDropSpots = validMoves;
    numberOfMoves += ((board.inHandPieces.Black.SilverGeneral > 0) +
                      (board.inHandPieces.Black.GoldGeneral > 0) +
                      (board.inHandPieces.Black.Bishop > 0) +
                      (board.inHandPieces.Black.Rook > 0)) *
                     (std::popcount<uint32_t>(legalDropSpots[TOP]) +
                      std::popcount<uint32_t>(legalDropSpots[MID]) +
                      std::popcount<uint32_t>(legalDropSpots[BOTTOM]));
  }

  return numberOfMoves;
}

void generateWhiteMoves(const Board& board,
                        const Bitboard& validMoves,
                        const Bitboard& attackedByEnemy,
                        Move* movesArray,
                        size_t offset) {
  Bitboard notPromoted = ~board[BB::Type::PROMOTED];
  Bitboard pieces, moves, ourAttacks;
  Bitboard occupied = board[BB::Type::ALL_WHITE] | board[BB::Type::ALL_BLACK];
  BitboardIterator movesIterator, iterator;
  Move move;
  Move* currentMove = movesArray + offset;
  // Pawn moves
  {
    pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] & notPromoted;
    moves = moveS(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + N;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        *currentMove = move;
        currentMove++;
      }
      // Not when forced promotion
      else if (move.to < WHITE_PAWN_LANCE_FORECED_PROMOTION_START) {
        move.promotion = 0;
        *currentMove = move;
        currentMove++;
      }
    }
  }
  // Knight moves
  {
    pieces = board[BB::Type::KNIGHT] & board[BB::Type::ALL_WHITE] & notPromoted;
    moves = moveS(moveSE(pieces)) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + N + NW;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        *currentMove = move;
        currentMove++;
      }
      // Not when forced promotion
      else if (move.to < WHITE_HORSE_FORCED_PROMOTION_START) {
        move.promotion = 0;
        *currentMove = move;
        currentMove++;
      }
    }
    moves = moveS(moveSW(pieces)) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + N + NE;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        *currentMove = move;
        currentMove++;
      }
      // Not when forced promotion
      else if (move.to < WHITE_HORSE_FORCED_PROMOTION_START) {
        move.promotion = 0;
        *currentMove = move;
        currentMove++;
      }
    }
  }
  // SilverGenerals moves
  {
    pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_WHITE] &
             notPromoted;
    moves = moveS(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + N;
      move.promotion = 0;
      *currentMove = move;
      currentMove++;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        *currentMove = move;
        currentMove++;
      }
    }
    moves = moveSE(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + NW;
      move.promotion = 0;
      *currentMove = move;
      currentMove++;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        *currentMove = move;
        currentMove++;
      }
    }
    moves = moveSW(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + NE;
      move.promotion = 0;
      *currentMove = move;
      currentMove++;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        *currentMove = move;
        currentMove++;
      }
    }
    moves = moveNE(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + SW;
      move.promotion = 0;
      *currentMove = move;
      currentMove++;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START ||
          move.from >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        *currentMove = move;
        currentMove++;
      }
    }
    moves = moveNW(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + SE;
      move.promotion = 0;
      *currentMove = move;
      currentMove++;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START ||
          move.from >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        *currentMove = move;
        currentMove++;
      }
    }
  }
  // GoldGenerals moves
  {
    pieces = (board[BB::Type::GOLD_GENERAL] |
              ((board[BB::Type::PAWN] | board[BB::Type::LANCE] |
                board[BB::Type::KNIGHT] | board[BB::Type::SILVER_GENERAL]) &
               board[BB::Type::PROMOTED])) &
             board[BB::Type::ALL_WHITE];
    moves = moveS(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + N;
      move.promotion = 0;
      *currentMove = move;
      currentMove++;
    }
    moves = moveSE(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + NW;
      move.promotion = 0;
      *currentMove = move;
      currentMove++;
    }
    moves = moveSW(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + NE;
      move.promotion = 0;
      *currentMove = move;
      currentMove++;
    }
    moves = moveE(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + E;
      move.promotion = 0;
      *currentMove = move;
      currentMove++;
    }
    moves = moveW(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + E;
      move.promotion = 0;
      *currentMove = move;
      currentMove++;
    }
    moves = moveN(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + S;
      move.promotion = 0;
      *currentMove = move;
      currentMove++;
    }
  }
  // Lances moves
  {
    pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] & notPromoted;
    iterator.Init(pieces);
    while (iterator.Next()) {
      move.from = iterator.GetCurrentSquare();
      moves = getFileAttacks(static_cast<Square>(move.from), occupied) &
              ~getRankMask(squareToRank(static_cast<Square>(move.from))) &
              validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        // Promotion
        if (move.to >= WHITE_PROMOTION_START) {
          move.promotion = 1;
          *currentMove = move;
          currentMove++;
        }
        // Not when forced promotion
        else if (move.to < WHITE_PAWN_LANCE_FORECED_PROMOTION_START) {
          move.promotion = 0;
          *currentMove = move;
          currentMove++;
        }
      }
    }
  }
  // Bishop moves
  {
    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] & notPromoted;
    iterator.Init(pieces);
    while (iterator.Next()) {
      move.from = iterator.GetCurrentSquare();
      moves = (getDiagRightAttacks(static_cast<Square>(move.from), occupied) |
               getDiagLeftAttacks(static_cast<Square>(move.from), occupied)) &
              validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.promotion = 0;
        *currentMove = move;
        currentMove++;
        // Promotion
        if (move.to >= WHITE_PROMOTION_START ||
            move.from >= WHITE_PROMOTION_START) {
          move.promotion = 1;
          *currentMove = move;
          currentMove++;
        }
      }
    }
  }
  // Rook moves
  {
    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] & notPromoted;
    iterator.Init(pieces);
    while (iterator.Next()) {
      move.from = iterator.GetCurrentSquare();
      moves = (getRankAttacks(static_cast<Square>(move.from), occupied) |
               getFileAttacks(static_cast<Square>(move.from), occupied)) &
              validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.promotion = 0;
        *currentMove = move;
        currentMove++;
        // Promotion
        if (move.to >= WHITE_PROMOTION_START ||
            move.from >= WHITE_PROMOTION_START) {
          move.promotion = 1;
          *currentMove = move;
          currentMove++;
        }
      }
    }
  }
  move.promotion = 0;
  // Horse moves
  {
    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
             board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.Next()) {
      move.from = iterator.GetCurrentSquare();
      moves = (getDiagRightAttacks(static_cast<Square>(move.from), occupied) |
               getDiagLeftAttacks(static_cast<Square>(move.from), occupied) |
               moveN(pieces) | moveE(pieces) | moveS(pieces) | moveW(pieces)) &
              validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        *currentMove = move;
        currentMove++;
      }
    }
  }
  // Dragon moves
  {
    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
             board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.Next()) {
      move.from = iterator.GetCurrentSquare();
      moves =
          (getRankAttacks(static_cast<Square>(move.from), occupied) |
           getFileAttacks(static_cast<Square>(move.from), occupied) |
           moveNW(pieces) | moveNE(pieces) | moveSE(pieces) | moveSW(pieces)) &
          validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        *currentMove = move;
        currentMove++;
      }
    }
  }
  // King moves
  {
    pieces = board[BB::Type::KING] & board[BB::Type::ALL_WHITE];
    iterator.Init(pieces);
    while (iterator.Next()) {
      move.from = iterator.GetCurrentSquare();
      moves =
          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
          ~attackedByEnemy & ~board[BB::Type::ALL_WHITE];
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        *currentMove = move;
        currentMove++;
      }
    }
  }
  // Generate Drop moves
  {
    Bitboard legalDropSpots;
    // Pawns
    if (board.inHandPieces.White.Pawn > 0) {
      legalDropSpots = validMoves;
      // Cannot drop on last rank
      legalDropSpots[BOTTOM] &= ~BOTTOM_RANK;
      // Cannot drop to give checkmate
      pieces = board[BB::Type::KING] & board[BB::Type::ALL_BLACK];
      // All valid enemy king moves
      moves =
          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
          ourAttacks;
      // If there is only one spot pawn cannot block it
      if (std::popcount<uint32_t>(moves[TOP]) +
              std::popcount<uint32_t>(moves[MID]) +
              std::popcount<uint32_t>(moves[BOTTOM]) ==
          1) {
        legalDropSpots &= ~moveN(moves);
      }
      // Cannot drop on file with other pawn
      Bitboard validFiles;
      for (int fileIdx = 0; fileIdx < BOARD_DIM; fileIdx++) {
        Bitboard file =
            getFileAttacks(static_cast<Square>(fileIdx), validFiles);
        if (file & board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
            notPromoted) {
          validFiles |= file;
        }
      }
      legalDropSpots &= validFiles;
      movesIterator.Init(legalDropSpots);
      move.from = WHITE_PAWN_DROP;
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        *currentMove = move;
        currentMove++;
      }
    }
    if (board.inHandPieces.White.Lance > 0) {
      legalDropSpots = validMoves;
      // Cannot drop on last rank
      legalDropSpots[BOTTOM] &= ~BOTTOM_RANK;
      movesIterator.Init(legalDropSpots);
      move.from = WHITE_KNIGHT_DROP;
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        *currentMove = move;
        currentMove++;
      }
    }
    if (board.inHandPieces.White.Knight > 0) {
      legalDropSpots = validMoves;
      // Cannot drop on last two ranks
      legalDropSpots[BOTTOM] &= TOP_RANK;
      movesIterator.Init(legalDropSpots);
      move.from = WHITE_LANCE_DROP;
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        *currentMove = move;
        currentMove++;
      }
    }
    legalDropSpots = validMoves;
    movesIterator.Init(legalDropSpots);
    while (movesIterator.Next()) {
      if (board.inHandPieces.White.SilverGeneral > 0) {
        move.from = WHITE_SILVER_GENERAL_DROP;
        move.to = movesIterator.GetCurrentSquare();
        *currentMove = move;
        currentMove++;
      }
      if (board.inHandPieces.White.GoldGeneral > 0) {
        move.from = WHITE_GOLD_GENERAL_DROP;
        move.to = movesIterator.GetCurrentSquare();
        *currentMove = move;
        currentMove++;
      }
      if (board.inHandPieces.White.Bishop > 0) {
        move.from = WHITE_BISHOP_DROP;
        move.to = movesIterator.GetCurrentSquare();
        *currentMove = move;
        currentMove++;
      }
      if (board.inHandPieces.White.Rook > 0) {
        move.from = WHITE_ROOK_DROP;
        move.to = movesIterator.GetCurrentSquare();
        *currentMove = move;
        currentMove++;
      }
    }
  }
}

void generateBlackMoves(const Board& board,
                        const Bitboard& validMoves,
                        const Bitboard& attackedByEnemy,
                        Move* movesArray,
                        size_t offset) {
  Bitboard notPromoted = ~board[BB::Type::PROMOTED];
  Bitboard pieces, moves, ourAttacks;
  Bitboard occupied = board[BB::Type::ALL_WHITE] | board[BB::Type::ALL_BLACK];
  BitboardIterator movesIterator, iterator;
  Move move;
  Move* currentMove = movesArray + offset;
  // Pawn moves
  {
    pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] & notPromoted;
    moves = moveN(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + S;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        *currentMove = move;
        currentMove++;
      }
      // Not when forced promotion
      else if (move.to > BLACK_PAWN_LANCE_FORCE_PROMOTION_END) {
        move.promotion = 0;
        *currentMove = move;
        currentMove++;
      }
    }
  }
  // Knight moves
  {
    pieces = board[BB::Type::KNIGHT] & board[BB::Type::ALL_BLACK] & notPromoted;
    moves = moveN(moveNE(pieces)) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + S + SW;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        *currentMove = move;
        currentMove++;
      }
      // Not when forced promotion
      else if (move.to > BLACK_HORSE_FORCED_PROMOTION_END) {
        move.promotion = 0;
        *currentMove = move;
        currentMove++;
      }
    }
    moves = moveN(moveNW(pieces)) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + S + SE;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        *currentMove = move;
        currentMove++;
      }
      // Not when forced promotion
      else if (move.to > BLACK_HORSE_FORCED_PROMOTION_END) {
        move.promotion = 0;
        *currentMove = move;
        currentMove++;
      }
    }
  }
  // SilverGenerals moves
  {
    pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_BLACK] &
             notPromoted;
    moves = moveN(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + S;
      move.promotion = 0;
      *currentMove = move;
      currentMove++;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        *currentMove = move;
        currentMove++;
      }
    }
    moves = moveNE(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + SW;
      move.promotion = 0;
      *currentMove = move;
      currentMove++;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        *currentMove = move;
        currentMove++;
      }
    }
    moves = moveNW(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + SE;
      move.promotion = 0;
      *currentMove = move;
      currentMove++;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        *currentMove = move;
        currentMove++;
      }
    }
    moves = moveSE(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + NW;
      move.promotion = 0;
      *currentMove = move;
      currentMove++;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END || move.from <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        *currentMove = move;
        currentMove++;
      }
    }
    moves = moveSW(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + NE;
      move.promotion = 0;
      *currentMove = move;
      currentMove++;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END || move.from <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        *currentMove = move;
        currentMove++;
      }
    }
  }
  // GoldGenerals moves
  {
    pieces = (board[BB::Type::GOLD_GENERAL] |
              ((board[BB::Type::PAWN] | board[BB::Type::LANCE] |
                board[BB::Type::KNIGHT] | board[BB::Type::SILVER_GENERAL]) &
               board[BB::Type::PROMOTED])) &
             board[BB::Type::ALL_BLACK];
    moves = moveN(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + S;
      move.promotion = 0;
      *currentMove = move;
      currentMove++;
    }
    moves = moveNE(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + SW;
      move.promotion = 0;
      *currentMove = move;
      currentMove++;
    }
    moves = moveNW(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + SE;
      move.promotion = 0;
      *currentMove = move;
      currentMove++;
    }
    moves = moveE(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + W;
      move.promotion = 0;
      *currentMove = move;
      currentMove++;
    }
    moves = moveW(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + E;
      move.promotion = 0;
      *currentMove = move;
      currentMove++;
    }
    moves = moveS(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + N;
      move.promotion = 0;
      *currentMove = move;
      currentMove++;
    }
  }
  // Lances moves
  {
    pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] & notPromoted;
    iterator.Init(pieces);
    while (iterator.Next()) {
      move.from = iterator.GetCurrentSquare();
      moves = getFileAttacks(static_cast<Square>(move.from), occupied) &
              getRankMask(squareToRank(static_cast<Square>(move.from))) &
              validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        // Promotion
        if (move.to <= BLACK_PROMOTION_END) {
          move.promotion = 1;
          *currentMove = move;
          currentMove++;
        }
        // Not when forced promotion
        else if (move.to > BLACK_PAWN_LANCE_FORCE_PROMOTION_END) {
          move.promotion = 0;
          *currentMove = move;
          currentMove++;
        }
      }
    }
  }
  // Bishop moves
  {
    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] & notPromoted;
    iterator.Init(pieces);
    while (iterator.Next()) {
      move.from = iterator.GetCurrentSquare();
      moves = (getDiagRightAttacks(static_cast<Square>(move.from), occupied) |
               getDiagLeftAttacks(static_cast<Square>(move.from), occupied)) &
              validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.promotion = 0;
        *currentMove = move;
        currentMove++;
        // Promotion
        if (move.to <= BLACK_PROMOTION_END ||
            move.from <= BLACK_PROMOTION_END) {
          move.promotion = 1;
          *currentMove = move;
          currentMove++;
        }
      }
    }
  }
  // Rook moves
  {
    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] & notPromoted;
    iterator.Init(pieces);
    while (iterator.Next()) {
      move.from = iterator.GetCurrentSquare();
      moves = (getRankAttacks(static_cast<Square>(move.from), occupied) |
               getFileAttacks(static_cast<Square>(move.from), occupied)) &
              validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.promotion = 0;
        *currentMove = move;
        currentMove++;
        // Promotion
        if (move.to <= BLACK_PROMOTION_END ||
            move.from <= BLACK_PROMOTION_END) {
          move.promotion = 1;
          *currentMove = move;
          currentMove++;
        }
      }
    }
  }
  move.promotion = 0;
  // Horse moves
  {
    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
             board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.Next()) {
      move.from = iterator.GetCurrentSquare();
      moves = (getDiagRightAttacks(static_cast<Square>(move.from), occupied) |
               getDiagLeftAttacks(static_cast<Square>(move.from), occupied) |
               moveN(pieces) | moveE(pieces) | moveS(pieces) | moveW(pieces)) &
              validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        *currentMove = move;
        currentMove++;
      }
    }
  }
  // Dragon moves
  {
    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
             board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.Next()) {
      move.from = iterator.GetCurrentSquare();
      moves =
          (getRankAttacks(static_cast<Square>(move.from), occupied) |
           getFileAttacks(static_cast<Square>(move.from), occupied) |
           moveNW(pieces) | moveNE(pieces) | moveSE(pieces) | moveSW(pieces)) &
          validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        *currentMove = move;
        currentMove++;
      }
    }
  }
  // King moves
  {
    pieces = board[BB::Type::KING] & board[BB::Type::ALL_BLACK];
    iterator.Init(pieces);
    while (iterator.Next()) {
      move.from = iterator.GetCurrentSquare();
      moves =
          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
          ~attackedByEnemy & ~board[BB::Type::ALL_BLACK];
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        *currentMove = move;
        currentMove++;
      }
    }
  }
  // Drop moves
  {
    Bitboard legalDropSpots;
    // Pawns
    if (board.inHandPieces.Black.Pawn > 0) {
      legalDropSpots = validMoves;
      // Cannot drop on last rank
      legalDropSpots[TOP] &= ~TOP_RANK;
      // Cannot drop to give checkmate
      pieces = board[BB::Type::KING] & board[BB::Type::ALL_WHITE];
      // All valid enemy king moves
      moves =
          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
          ourAttacks;
      // If there is only one spot pawn cannot block it
      if (std::popcount<uint32_t>(moves[TOP]) +
              std::popcount<uint32_t>(moves[MID]) +
              std::popcount<uint32_t>(moves[BOTTOM]) ==
          1) {
        legalDropSpots &= ~moveS(moves);
      }
      // Cannot drop on file with other pawn
      Bitboard validFiles;
      for (int fileIdx = 0; fileIdx < BOARD_DIM; fileIdx++) {
        Bitboard file =
            getFileAttacks(static_cast<Square>(fileIdx), validFiles);
        if (file & board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
            notPromoted) {
          validFiles |= file;
        }
      }
      legalDropSpots &= validFiles;
      move.from = BLACK_PAWN_DROP;
      movesIterator.Init(legalDropSpots);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        *currentMove = move;
        currentMove++;
      }
    }
    if (board.inHandPieces.Black.Lance > 0) {
      legalDropSpots = validMoves;
      // Cannot drop on last rank
      legalDropSpots[TOP] &= ~TOP_RANK;
      move.from = BLACK_LANCE_DROP;
      movesIterator.Init(legalDropSpots);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        *currentMove = move;
        currentMove++;
      }
    }
    if (board.inHandPieces.Black.Knight > 0) {
      legalDropSpots = validMoves;
      // Cannot drop on last two ranks
      legalDropSpots[TOP] &= TOP_RANK;
      move.from = BLACK_KNIGHT_DROP;
      movesIterator.Init(legalDropSpots);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        *currentMove = move;
        currentMove++;
      }
    }
    legalDropSpots = validMoves;
    movesIterator.Init(legalDropSpots);
    while (movesIterator.Next()) {
      if (board.inHandPieces.Black.SilverGeneral > 0) {
        move.from = BLACK_SILVER_GENERAL_DROP;
        move.to = movesIterator.GetCurrentSquare();
        *currentMove = move;
        currentMove++;
      }
      if (board.inHandPieces.Black.GoldGeneral > 0) {
        move.from = BLACK_GOLD_GENERAL_DROP;
        move.to = movesIterator.GetCurrentSquare();
        *currentMove = move;
        currentMove++;
      }
      if (board.inHandPieces.Black.Bishop > 0) {
        move.from = BLACK_BISHOP_DROP;
        move.to = movesIterator.GetCurrentSquare();
        *currentMove = move;
        currentMove++;
      }
      if (board.inHandPieces.Black.Rook > 0) {
        move.from = BLACK_ROOK_DROP;
        move.to = movesIterator.GetCurrentSquare();
        *currentMove = move;
        currentMove++;
      }
    }
  }
}

std::array<std::vector<std::pair<int, bool>>, BOARD_SIZE + 14> getAllLegalMoves(
    const Board& board,
    bool isWhite) {
  std::array<std::vector<std::pair<int, bool>>, BOARD_SIZE + 14> result;
  Bitboard validMoves, attackedByEnemy;
  if (isWhite) {
    size_t movesCount = countWhiteMoves(board, validMoves, attackedByEnemy);
    Move* moves = new Move[movesCount];
    generateWhiteMoves(board, validMoves, attackedByEnemy, moves, 0);
    for (int i = 0; i < movesCount; i++) {
      std::vector<std::pair<int, bool>>& movesFromSquare =
          result[moves[i].from];
      if (moves[i].promotion) {
        movesFromSquare.back().second = true;
      }
      movesFromSquare.push_back({moves[i].to, false});
    }
  } else {
    size_t movesCount = countBlackMoves(board, validMoves, attackedByEnemy);
    Move* moves = new Move[movesCount];
    generateBlackMoves(board, validMoves, attackedByEnemy, moves, 0);
    for (int i = 0; i < movesCount; i++) {
      std::vector<std::pair<int, bool>>& movesFromSquare =
          result[moves[i].from];
      if (moves[i].promotion) {
        movesFromSquare.back().second = true;
      }
      movesFromSquare.push_back({moves[i].to, false});
    }
  }
  return result;
}