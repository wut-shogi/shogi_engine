#include "LookUpTables.h"
#include "MoveGen.h"
#include "MoveGenHelpers.h"
namespace shogi {
namespace engine {
//////////////////////////
// W pinned zapisujesz spinowane i pinuj¹ce (potem to siê da odzdzieliæ
// kolorami). Dla spinowanych figur ich ruchy to ruch & odpowiedni slajd od
// króla (np file dla pionków) & pinuj¹ce
// Teoretycznie zosta³o tylko generatemoves do przerobienia
RUNTYPE void getWhitePiecesInfo(const Board& board,
                                Bitboard& outPinned,
                                Bitboard& outValidMoves,
                                Bitboard& outAttackedByEnemy) {
  Bitboard king = board[BB::Type::KING] & board[BB::Type::ALL_WHITE];
  Bitboard pieces, moves, attacks, attacksFull, mask, potentialPin, pinned,
      attackedByEnemy, enemyCheckingPieces, enemySlidingChecksPaths;
  BitboardIterator iterator;
  Square square;

  // Non Sliding pieces
  // King
  pieces = board[BB::Type::KING] & board[BB::Type::ALL_BLACK];
  enemyCheckingPieces |=
      (moveNE(king) | moveN(king) | moveNW(king) | moveE(king) | moveW(pieces) |
       moveSE(king) | moveS(king) | moveSW(king)) &
      pieces;
  attackedByEnemy |= moveNE(pieces) | moveN(pieces) | moveNW(pieces) |
                     moveE(pieces) | moveW(pieces) | moveSE(pieces) |
                     moveS(pieces) | moveSW(pieces);
  // Pawns
  pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
           ~board[BB::Type::PROMOTED];
  enemyCheckingPieces |= moveS(king) & pieces;
  attackedByEnemy |= moveN(pieces);
  // Knights
  pieces = board[BB::Type::KNIGHT] & board[BB::Type::ALL_BLACK] &
           ~board[BB::Type::PROMOTED];
  enemyCheckingPieces |= moveS(moveSE(king) | moveSW(king)) & pieces;
  attackedByEnemy |= moveN(moveNE(pieces) | moveNW(pieces));
  // Silver generals
  pieces = board[BB::Type::SILVER_GENERAL] &
           board[BB::Type::ALL_BLACK] & ~board[BB::Type::PROMOTED];
  enemyCheckingPieces |= (moveSE(king) | moveS(king) | moveSW(king) |
                          moveNE(king) | moveNW(king)) &
                         pieces;
  attackedByEnemy |= moveNE(pieces) | moveN(pieces) | moveNW(pieces) |
                     moveSE(pieces) | moveSW(pieces);
  // Gold generals
  pieces = (board[BB::Type::GOLD_GENERAL] |
            ((board[BB::Type::PAWN] | board[BB::Type::KNIGHT] |
              board[BB::Type::SILVER_GENERAL] | board[BB::Type::LANCE]) &
             board[BB::Type::PROMOTED])) &
           board[BB::Type::ALL_BLACK];
  enemyCheckingPieces |= (moveSE(king) | moveS(king) | moveSW(king) |
                          moveE(king) | moveW(king) | moveN(king)) &
                         pieces;
  attackedByEnemy |= moveNE(pieces) | moveN(pieces) | moveNW(pieces) |
                     moveE(pieces) | moveW(pieces) | moveS(pieces);
  // Horse (non sliding part)
  pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
           board[BB::Type::PROMOTED];
  enemyCheckingPieces |=
      (moveN(king) | moveE(king) | moveS(king) | moveW(king)) & pieces;
  attackedByEnemy |=
      moveN(pieces) | moveE(pieces) | moveS(pieces) | moveW(pieces);
  // Dragon (non sliding part)
  pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
           board[BB::Type::PROMOTED];
  enemyCheckingPieces |=
      (moveNW(king) | moveNE(king) | moveSE(king) | moveSW(king)) & pieces;
  attackedByEnemy |=
      moveNW(pieces) | moveNE(pieces) | moveSE(pieces) | moveSW(pieces);

  iterator.Init(king);
  iterator.Next();
  Square kingSquare = iterator.GetCurrentSquare();
  Bitboard occupied = board[BB::Type::ALL_BLACK] | board[BB::Type::ALL_WHITE];
  // Lance
  {
    pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] &
             ~board[BB::Type::PROMOTED];
    enemyCheckingPieces |=
        LookUpTables::getFileAttacks(kingSquare, occupied) &
        ~LookUpTables::getRankMask(squareToRank(kingSquare)) & pieces;
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      mask = LookUpTables::getRankMask(squareToRank(square));
      attacksFull = LookUpTables::getFileAttacks(square, occupied) & mask;
      attackedByEnemy |= attacksFull;
      if (!(attacksFull & king)) {
        potentialPin = attacksFull & board[BB::ALL_WHITE];
        attacks =
            LookUpTables::getFileAttacks(square, occupied & ~potentialPin);
        if (attacks & king) {
          pinned |= potentialPin;
          pinned |= Bitboard(square);
        }
      } else {
        enemySlidingChecksPaths |= attacksFull;
        attackedByEnemy |=
            LookUpTables::getFileAttacks(square, occupied & ~king) & mask;
      }
    }
  }

  // Rook and dragon
  {
    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK];
    enemyCheckingPieces |=
        (LookUpTables::getRankAttacks(kingSquare, occupied) |
         LookUpTables::getFileAttacks(kingSquare, occupied)) &
        pieces;
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      // Check if king is in check without white pieces
      // We have to check all 4 directions
      // left-right
      attacksFull = LookUpTables::getRankAttacks(square, occupied);
      attackedByEnemy |= attacksFull;
      mask = LookUpTables::getFileMask(squareToFile(square));
      // left
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & mask;
        attacks =
            LookUpTables::getRankAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
          pinned |= Bitboard(square);
        }
      } else {
        enemySlidingChecksPaths |= attacksFull & mask;
        attackedByEnemy |=
            LookUpTables::getRankAttacks(square, occupied & ~king) & mask;
      }
      // right
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & ~mask;
        attacks =
            LookUpTables::getRankAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
          pinned |= Bitboard(square);
        }
      } else {
        enemySlidingChecksPaths |= attacksFull & ~mask;
        attackedByEnemy |=
            LookUpTables::getRankAttacks(square, occupied & ~king) & ~mask;
      }
      // up-down
      attacksFull = LookUpTables::getFileAttacks(square, occupied);
      attackedByEnemy |= attacksFull;
      mask = LookUpTables::getRankMask(squareToRank(square));
      // up
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & mask;
        attacks =
            LookUpTables::getFileAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
          pinned |= Bitboard(square);
        }
      } else {
        enemySlidingChecksPaths |= attacksFull & mask;
        attackedByEnemy |=
            LookUpTables::getFileAttacks(square, occupied & ~king) & mask;
      }
      // down
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & ~mask;
        attacks =
            LookUpTables::getFileAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
          pinned |= Bitboard(square);
        }
      } else {
        enemySlidingChecksPaths |= attacksFull & ~mask;
        attackedByEnemy |=
            LookUpTables::getFileAttacks(square, occupied & ~king) & ~mask;
      }
    }
  }

  // Bishop and horse pins
  {
    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK];
    enemyCheckingPieces |=
        (LookUpTables::getDiagRightAttacks(kingSquare, occupied) |
         LookUpTables::getDiagLeftAttacks(kingSquare, occupied)) &
        pieces;
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      // We have to check all 4 directions
      // right diag
      attacksFull = LookUpTables::getDiagRightAttacks(square, occupied);
      attackedByEnemy |= attacksFull;
      mask = ~LookUpTables::getFileMask(squareToFile(square)) &
             LookUpTables::getRankMask(squareToRank(square));
      // SW
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & mask;
        attacks =
            LookUpTables::getDiagRightAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
          pinned |= Bitboard(square);
        }
      } else {
        enemySlidingChecksPaths |= attacksFull & mask;
        attackedByEnemy |=
            LookUpTables::getDiagRightAttacks(square, occupied & ~king) & mask;
      }
      // NE
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & ~mask;
        attacks =
            LookUpTables::getDiagRightAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
          pinned |= Bitboard(square);
        }
      } else {
        enemySlidingChecksPaths |= attacksFull & ~mask;
        attackedByEnemy |=
            LookUpTables::getDiagRightAttacks(square, occupied & ~king) & ~mask;
      }
      // left diag
      attacksFull = LookUpTables::getDiagLeftAttacks(square, occupied);
      attackedByEnemy |= attacksFull;
      mask = LookUpTables::getFileMask(squareToFile(square)) &
             LookUpTables::getRankMask(squareToRank(square));
      // NW
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & mask;
        attacks =
            LookUpTables::getDiagLeftAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
          pinned |= Bitboard(square);
        }
      } else {
        enemySlidingChecksPaths |= attacksFull & mask;
        attackedByEnemy |=
            LookUpTables::getDiagLeftAttacks(square, occupied & ~king) & mask;
      }
      // SE
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & ~mask;
        attacks =
            LookUpTables::getDiagLeftAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
          pinned |= Bitboard(square);
        }
      } else {
        enemySlidingChecksPaths |= attacksFull & ~mask;
        attackedByEnemy |=
            LookUpTables::getDiagLeftAttacks(square, occupied & ~king) & ~mask;
      }
    }
  }

  int numberOfCheckingPieces = popcount(enemyCheckingPieces[TOP]) +
                               popcount(enemyCheckingPieces[MID]) +
                               popcount(enemyCheckingPieces[BOTTOM]);
  // If more then one piece is checking the king and king cannot move its mate
  if (numberOfCheckingPieces == 1) {
    // if king is checked by exactly one piece legal moves can also be block
    // sliding check or capture a checking piece
    outValidMoves = enemyCheckingPieces | (enemySlidingChecksPaths & ~king);
  } else if (numberOfCheckingPieces == 0) {
    // If there is no checks all moves are valid (you cannot capture your own
    // piece)
    outValidMoves = ~board[BB::Type::ALL_WHITE];
  } else {
    outValidMoves = {0, 0, 0};
  }
  outPinned = pinned;
  outAttackedByEnemy = attackedByEnemy;
}

RUNTYPE void getBlackPiecesInfo(const Board& board,
                                Bitboard& outPinned,
                                Bitboard& outValidMoves,
                                Bitboard& outAttackedByEnemy) {
  Bitboard king = board[BB::Type::KING] & board[BB::Type::ALL_BLACK];
  Bitboard pieces, moves, attacks, attacksFull, mask, potentialPin, pinned,
      attackedByEnemy, enemyCheckingPieces, enemySlidingChecksPaths;
  BitboardIterator iterator;
  Square square;

  // Non Sliding pieces
  // King
  pieces = board[BB::Type::KING] & board[BB::Type::ALL_WHITE];
  enemyCheckingPieces |=
      (moveNE(king) | moveN(king) | moveNW(king) | moveE(king) | moveW(pieces) |
       moveSE(king) | moveS(king) | moveSW(king)) &
      pieces;
  attackedByEnemy |= moveNE(pieces) | moveN(pieces) | moveNW(pieces) |
                     moveE(pieces) | moveW(pieces) | moveSE(pieces) |
                     moveS(pieces) | moveSW(pieces);
  // Pawns
  pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
           ~board[BB::Type::PROMOTED];
  enemyCheckingPieces |= moveN(king) & pieces;
  attackedByEnemy |= moveS(pieces);
  // Knights
  pieces = board[BB::Type::KNIGHT] & board[BB::Type::ALL_WHITE] &
           ~board[BB::Type::PROMOTED];
  enemyCheckingPieces |= moveN(moveNE(king) | moveNW(king)) & pieces;
  attackedByEnemy |= moveS(moveSE(pieces) | moveSW(pieces));
  // Silver generals
  pieces = board[BB::Type::SILVER_GENERAL] &
           board[BB::Type::ALL_WHITE] & ~board[BB::Type::PROMOTED];
  enemyCheckingPieces |= (moveNE(king) | moveN(king) | moveNW(king) |
                          moveSE(king) | moveSW(king)) &
                         pieces;
  attackedByEnemy |= moveSE(pieces) | moveS(pieces) | moveSW(pieces) |
                     moveNE(pieces) | moveNW(pieces);
  // gold generals
  pieces = (board[BB::Type::GOLD_GENERAL] |
            ((board[BB::Type::PAWN] | board[BB::Type::KNIGHT] |
              board[BB::Type::SILVER_GENERAL] | board[BB::Type::LANCE]) &
             board[BB::Type::PROMOTED])) &
           board[BB::Type::ALL_WHITE];
  enemyCheckingPieces |= (moveNE(king) | moveN(king) | moveNW(king) |
                          moveE(king) | moveW(king) | moveS(king)) &
                         pieces;
  attackedByEnemy |= moveSE(pieces) | moveS(pieces) | moveSW(pieces) |
                     moveE(pieces) | moveW(pieces) | moveN(pieces);
  // Horse (non sliding part)
  pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
           board[BB::Type::PROMOTED];
  enemyCheckingPieces |=
      (moveN(king) | moveE(king) | moveS(king) | moveW(king)) & pieces;
  attackedByEnemy |=
      moveN(pieces) | moveE(pieces) | moveS(pieces) | moveW(pieces);
  // Dragon (non sldiing part)
  pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
           board[BB::Type::PROMOTED];
  enemyCheckingPieces |=
      (moveNW(king) | moveNE(king) | moveSE(king) | moveSW(king)) & pieces;
  attackedByEnemy |=
      moveNW(pieces) | moveNE(pieces) | moveSE(pieces) | moveSW(pieces);
  // Sliding pieces
  iterator.Init(king);
  iterator.Next();
  Square kingSquare = iterator.GetCurrentSquare();
  Bitboard occupied = board[BB::Type::ALL_BLACK] | board[BB::Type::ALL_WHITE];
  // Lance
  {
    pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] &
             ~board[BB::Type::PROMOTED];
    enemyCheckingPieces |= LookUpTables::getFileAttacks(kingSquare, occupied) &
                           LookUpTables::getRankMask(squareToRank(kingSquare)) &
                           pieces;
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      mask = ~LookUpTables::getRankMask(squareToRank(square));
      attacksFull = LookUpTables::getFileAttacks(square, occupied) & mask;
      attackedByEnemy |= attacksFull;
      if (!(attacksFull & king)) {
        potentialPin = attacksFull & board[BB::ALL_BLACK];
        attacks =
            LookUpTables::getFileAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
          pinned |= Bitboard(square);
        }
      } else {
        enemySlidingChecksPaths |= attacksFull;
        attackedByEnemy |=
            LookUpTables::getFileAttacks(square, occupied & ~king) & mask;
      }
    }
  }

  // Rook and dragon
  {
    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE];
    enemyCheckingPieces |=
        (LookUpTables::getRankAttacks(kingSquare, occupied) |
         LookUpTables::getFileAttacks(kingSquare, occupied)) &
        pieces;
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      // Check if king is in check without white pieces
      // We have to check all 4 directions
      // left-right
      attacksFull = LookUpTables::getRankAttacks(square, occupied);
      attackedByEnemy |= attacksFull;
      mask = LookUpTables::getFileMask(squareToFile(square));
      // left
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & mask;
        attacks =
            LookUpTables::getRankAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
          pinned |= Bitboard(square);
        }
      } else {
        enemySlidingChecksPaths |= attacksFull & mask;
        attackedByEnemy |=
            LookUpTables::getRankAttacks(square, occupied & ~king) & mask;
      }
      // right
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & ~mask;
        attacks =
            LookUpTables::getRankAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
          pinned |= Bitboard(square);
        }
      } else {
        enemySlidingChecksPaths |= attacksFull & ~mask;
        attackedByEnemy |=
            LookUpTables::getRankAttacks(square, occupied & ~king) & ~mask;
      }
      // up-down
      attacksFull = LookUpTables::getFileAttacks(square, occupied);
      attackedByEnemy |= attacksFull;
      mask = LookUpTables::getRankMask(squareToRank(square));
      // up
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & mask;
        attacks =
            LookUpTables::getFileAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
          pinned |= Bitboard(square);
        }
      } else {
        enemySlidingChecksPaths |= attacksFull & mask;
        attackedByEnemy |=
            LookUpTables::getFileAttacks(square, occupied & ~king) & mask;
      }
      // down
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & ~mask;
        attacks =
            LookUpTables::getFileAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
          pinned |= Bitboard(square);
        }
      } else {
        enemySlidingChecksPaths |= attacksFull & ~mask;
        attackedByEnemy |=
            LookUpTables::getFileAttacks(square, occupied & ~king) & ~mask;
      }
    }
  }

  // Bishop and horse pins
  {
    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE];
    enemyCheckingPieces |=
        (LookUpTables::getDiagRightAttacks(kingSquare, occupied) |
         LookUpTables::getDiagLeftAttacks(kingSquare, occupied)) &
        pieces;
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      // Check if king is in check without white pieces
      // We have to check all 4 directions
      // right diag
      attacksFull = LookUpTables::getDiagRightAttacks(square, occupied);
      attackedByEnemy |= attacksFull;
      mask = ~LookUpTables::getFileMask(squareToFile(square)) &
             LookUpTables::getRankMask(squareToRank(square));
      // SW
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & mask;
        attacks =
            LookUpTables::getDiagRightAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
          pinned |= Bitboard(square);
        }
      } else {
        enemySlidingChecksPaths |= attacksFull & mask;
        attackedByEnemy |=
            LookUpTables::getDiagRightAttacks(square, occupied & ~king) & mask;
      }
      // NE
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & ~mask;
        attacks =
            LookUpTables::getDiagRightAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
          pinned |= Bitboard(square);
        }
      } else {
        enemySlidingChecksPaths |= attacksFull & ~mask;
        attackedByEnemy |=
            LookUpTables::getDiagRightAttacks(square, occupied & ~king) & ~mask;
      }
      // left diag
      attacksFull = LookUpTables::getDiagLeftAttacks(square, occupied);
      attackedByEnemy |= attacksFull;
      mask = LookUpTables::getFileMask(squareToFile(square)) &
             LookUpTables::getRankMask(squareToRank(square));
      // NW
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & mask;
        attacks =
            LookUpTables::getDiagLeftAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
          pinned |= Bitboard(square);
        }
      } else {
        enemySlidingChecksPaths |= attacksFull & mask;
        attackedByEnemy |=
            LookUpTables::getDiagLeftAttacks(square, occupied & ~king) & mask;
      }
      // SE
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & ~mask;
        attacks =
            LookUpTables::getDiagLeftAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
          pinned |= Bitboard(square);
        }
      } else {
        enemySlidingChecksPaths |= attacksFull & ~mask;
        attackedByEnemy |=
            LookUpTables::getDiagLeftAttacks(square, occupied & ~king) & ~mask;
      }
    }
  }

  int numberOfCheckingPieces = popcount(enemyCheckingPieces[TOP]) +
                               popcount(enemyCheckingPieces[MID]) +
                               popcount(enemyCheckingPieces[BOTTOM]);
  // If more then one piece is checking the king and king cannot move its mate
  if (numberOfCheckingPieces == 1) {
    // if king is checked by exactly one piece legal moves can also be block
    // sliding check or capture a checking piece
    outValidMoves = enemyCheckingPieces | (enemySlidingChecksPaths & ~king);
  } else if (numberOfCheckingPieces == 0) {
    // If there is no checks all moves are valid (you cannot capture your own
    // piece)
    outValidMoves = ~board[BB::Type::ALL_BLACK];
  } else {
    outValidMoves = {0, 0, 0};
  }
  outPinned = pinned;
  outAttackedByEnemy = attackedByEnemy;
}

RUNTYPE uint32_t countWhiteMoves(const Board& board,
                                 Bitboard& pinned,
                                 Bitboard& validMoves,
                                 Bitboard& attackedByEnemy) {
  Bitboard pieces, moves, ourAttacks;
  Bitboard occupied = board[BB::Type::ALL_WHITE] | board[BB::Type::ALL_BLACK];
  BitboardIterator iterator;
  Square square;
  uint32_t numberOfMoves = 0;
  Bitboard pinnedPieces = pinned & board[BB::Type::ALL_WHITE];
  Bitboard pinningPieces = pinned & board[BB::Type::ALL_BLACK];
  Bitboard kingRayRank, kingRayFile, kingRayDiagRight, kingRayDiagLeft;

  // King moves
  {
    pieces = board[BB::Type::KING] & board[BB::Type::ALL_WHITE];
    moves = moveNE(pieces) | moveN(pieces) | moveNW(pieces) | moveE(pieces) |
            moveW(pieces) | moveSE(pieces) | moveS(pieces) | moveSW(pieces);
    moves &= ~attackedByEnemy & ~board[BB::Type::ALL_WHITE];
    numberOfMoves +=
        popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);

    if (pinnedPieces) {
      iterator.Init(pieces);
      iterator.Next();
      Square kingSquare = iterator.GetCurrentSquare();
      Bitboard empty = {0, 0, 0};
      kingRayRank = LookUpTables::getRankAttacks(kingSquare, empty);
      kingRayFile = LookUpTables::getFileAttacks(kingSquare, empty);
      kingRayDiagRight = LookUpTables::getDiagRightAttacks(kingSquare, empty);
      kingRayDiagLeft = LookUpTables::getDiagLeftAttacks(kingSquare, empty);
    }
  }

  // Pawn moves
  {
    pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
             ~board[BB::Type::PROMOTED];
    moves = moveS((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayFile)) &
            validMoves;
    ourAttacks |= moves;
    numberOfMoves += popcount(moves[TOP]) + popcount(moves[MID]) +
                     popcount(moves[BOTTOM] & ~BOTTOM_RANK) * 2 +  // promotions
                     popcount(moves[BOTTOM] & BOTTOM_RANK);  // forced promotion
  }

  // Knight moves
  {
    pieces = ~pinnedPieces & board[BB::Type::KNIGHT] &
             board[BB::Type::ALL_WHITE] & ~board[BB::Type::PROMOTED];
    moves = moveS(moveSE(pieces)) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += popcount(moves[TOP]) + popcount(moves[MID]) +
                     popcount(moves[BOTTOM] & TOP_RANK) * 2 +  // promotions
                     popcount(moves[BOTTOM] & ~TOP_RANK);  // forced promotion
    moves = moveS(moveSW(pieces)) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += popcount(moves[TOP]) + popcount(moves[MID]) +
                     popcount(moves[BOTTOM] & TOP_RANK) * 2 +  // promotions
                     popcount(moves[BOTTOM] & ~TOP_RANK);  // forced promotion
  }

  // SilverGenerals moves
  {
    pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_WHITE] &
             ~board[BB::Type::PROMOTED];
    moves = moveS((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayFile)) &
            validMoves;
    ourAttacks |= moves;
    numberOfMoves += popcount(moves[TOP]) + popcount(moves[MID]) +
                     popcount(moves[BOTTOM]) * 2;  // promotions
    moves = moveSE((pieces & ~pinnedPieces) |
                   (pieces & pinnedPieces & kingRayDiagLeft)) &
            validMoves;
    ourAttacks |= moves;
    numberOfMoves += popcount(moves[TOP]) + popcount(moves[MID]) +
                     popcount(moves[BOTTOM]) * 2;  // promotions
    moves = moveSW((pieces & ~pinnedPieces) |
                   (pieces & pinnedPieces & kingRayDiagRight)) &
            validMoves;
    ourAttacks |= moves;
    numberOfMoves += popcount(moves[TOP]) + popcount(moves[MID]) +
                     popcount(moves[BOTTOM]) * 2;  // promotions
    moves = moveNE((pieces & ~pinnedPieces) |
                   (pieces & pinnedPieces & kingRayDiagRight)) &
            validMoves;
    ourAttacks |= moves;
    numberOfMoves += popcount(moves[TOP]) +
                     popcount(moves[MID] & BOTTOM_RANK) *
                         2 +  // promotion when starting from promotion zone
                     popcount(moves[MID] & ~BOTTOM_RANK) +
                     popcount(moves[BOTTOM]) * 2;  // promotions
    moves = moveNW((pieces & ~pinnedPieces) |
                   (pieces & pinnedPieces & kingRayDiagLeft)) &
            validMoves;
    ourAttacks |= moves;
    numberOfMoves += popcount(moves[TOP]) +
                     popcount(moves[MID] & BOTTOM_RANK) *
                         2 +  // promotion when starting from promotion zone
                     popcount(moves[MID] & ~BOTTOM_RANK) +
                     popcount(moves[BOTTOM]) * 2;  // promotions
  }

  // GoldGenerals moves
  {
    pieces = (board[BB::Type::GOLD_GENERAL] |
              ((board[BB::Type::PAWN] | board[BB::Type::LANCE] |
                board[BB::Type::KNIGHT] | board[BB::Type::SILVER_GENERAL]) &
               board[BB::Type::PROMOTED])) &
             board[BB::Type::ALL_WHITE];
    moves = moveS((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayFile)) &
            validMoves;
    ourAttacks |= moves;
    numberOfMoves +=
        popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
    moves = moveSE((pieces & ~pinnedPieces) |
                   (pieces & pinnedPieces & kingRayDiagLeft)) &
            validMoves;
    ourAttacks |= moves;
    numberOfMoves +=
        popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
    moves = moveSW((pieces & ~pinnedPieces) |
                   (pieces & pinnedPieces & kingRayDiagRight)) &
            validMoves;
    ourAttacks |= moves;
    numberOfMoves +=
        popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
    moves = moveE((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayRank)) &
            validMoves;
    ourAttacks |= moves;
    numberOfMoves +=
        popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
    moves = moveW((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayRank)) &
            validMoves;
    ourAttacks |= moves;
    numberOfMoves +=
        popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
    moves = moveN((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayFile)) &
            validMoves;
    ourAttacks |= moves;
    numberOfMoves +=
        popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
  }

  // Lance moves
  {
    pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] &
             ~board[BB::Type::PROMOTED];
    iterator.Init((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayFile));
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      moves = LookUpTables::getFileAttacks(square, occupied) &
              ~LookUpTables::getRankMask(squareToRank(square)) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) +
          popcount(moves[BOTTOM] & ~BOTTOM_RANK) * 2 +  // promotions
          popcount(moves[BOTTOM] & BOTTOM_RANK);        // forced promotion
    }
  }

  // Bishop moves
  {
    pieces = ~pinnedPieces & board[BB::Type::BISHOP] &
             board[BB::Type::ALL_WHITE] & ~board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      moves = (LookUpTables::getDiagRightAttacks(square, occupied) |
               LookUpTables::getDiagLeftAttacks(square, occupied)) &
              validMoves;
      ourAttacks |= moves;
      if (square >= WHITE_PROMOTION_START) {  // Starting from promotion zone
        numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                          popcount(moves[BOTTOM])) *
                         2;
      } else {
        numberOfMoves += popcount(moves[TOP]) + popcount(moves[MID]) +
                         popcount(moves[BOTTOM]) * 2;  // end in promotion Zone
      }
    }

    pieces = pinnedPieces & board[BB::Type::BISHOP] &
             board[BB::Type::ALL_WHITE] & ~board[BB::Type::PROMOTED];
    if (pieces) {
      iterator.Init(pieces & kingRayDiagLeft);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        moves = LookUpTables::getDiagLeftAttacks(square, occupied) & validMoves;
        ourAttacks |= moves;
        if (square >= WHITE_PROMOTION_START) {  // Starting from promotion zone
          numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                            popcount(moves[BOTTOM])) *
                           2;
        } else {
          numberOfMoves +=
              popcount(moves[TOP]) + popcount(moves[MID]) +
              popcount(moves[BOTTOM]) * 2;  // end in promotion Zone
        }
      }
      iterator.Init(pieces & kingRayDiagRight);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        moves =
            LookUpTables::getDiagRightAttacks(square, occupied) & validMoves;
        ourAttacks |= moves;
        if (square >= WHITE_PROMOTION_START) {  // Starting from promotion zone
          numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                            popcount(moves[BOTTOM])) *
                           2;
        } else {
          numberOfMoves +=
              popcount(moves[TOP]) + popcount(moves[MID]) +
              popcount(moves[BOTTOM]) * 2;  // end in promotion Zone
        }
      }
    }
  }

  // Rook moves
  {
    pieces = ~pinnedPieces & board[BB::Type::ROOK] &
             board[BB::Type::ALL_WHITE] & ~board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      moves = (LookUpTables::getRankAttacks(square, occupied) |
               LookUpTables::getFileAttacks(square, occupied)) &
              validMoves;
      ourAttacks |= moves;
      if (square >= WHITE_PROMOTION_START) {  // Starting from promotion zone
        numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                          popcount(moves[BOTTOM])) *
                         2;
      } else {
        numberOfMoves += popcount(moves[TOP]) + popcount(moves[MID]) +
                         popcount(moves[BOTTOM]) * 2;  // end in promotion Zone
      }
    }

    pieces = pinnedPieces & board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
             ~board[BB::Type::PROMOTED];
    if (pieces) {
      iterator.Init(pieces & kingRayRank);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        moves = LookUpTables::getRankAttacks(square, occupied) & validMoves;
        ourAttacks |= moves;
        if (square >= WHITE_PROMOTION_START) {  // Starting from promotion zone
          numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                            popcount(moves[BOTTOM])) *
                           2;
        } else {
          numberOfMoves +=
              popcount(moves[TOP]) + popcount(moves[MID]) +
              popcount(moves[BOTTOM]) * 2;  // end in promotion Zone
        }
      }
      iterator.Init(pieces & kingRayFile);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        moves = LookUpTables::getFileAttacks(square, occupied) & validMoves;
        ourAttacks |= moves;
        if (square >= WHITE_PROMOTION_START) {  // Starting from promotion zone
          numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                            popcount(moves[BOTTOM])) *
                           2;
        } else {
          numberOfMoves +=
              popcount(moves[TOP]) + popcount(moves[MID]) +
              popcount(moves[BOTTOM]) * 2;  // end in promotion Zone
        }
      }
    }
  }

  // Horse moves
  {
    pieces = ~pinnedPieces & board[BB::Type::BISHOP] &
             board[BB::Type::ALL_WHITE] & board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      Bitboard horse = Bitboard(square);
      moves = (LookUpTables::getDiagRightAttacks(square, occupied) |
               LookUpTables::getDiagLeftAttacks(square, occupied) |
               moveN(horse) | moveE(horse) | moveS(horse) | moveW(horse)) &
              validMoves;
      ourAttacks |= moves;
      numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                        popcount(moves[BOTTOM]));
    }

    pieces = pinnedPieces & board[BB::Type::BISHOP] &
             board[BB::Type::ALL_WHITE] & board[BB::Type::PROMOTED];
    if (pieces) {
      iterator.Init(pieces & kingRayDiagLeft);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        moves = LookUpTables::getDiagLeftAttacks(square, occupied) & validMoves;
        ourAttacks |= moves;
        numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                          popcount(moves[BOTTOM]));
      }
      iterator.Init(pieces & kingRayDiagRight);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        moves =
            LookUpTables::getDiagRightAttacks(square, occupied) & validMoves;
        ourAttacks |= moves;
        numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                          popcount(moves[BOTTOM]));
      }
      iterator.Init(pieces & kingRayFile);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        Bitboard horse(square);
        moves = (moveN(horse) | moveS(horse)) & validMoves;
        ourAttacks |= moves;
        numberOfMoves += popcount(moves[TOP]) + popcount(moves[MID]) +
                         popcount(moves[BOTTOM]);
      }
      iterator.Init(pieces & kingRayRank);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        Bitboard horse(square);
        moves = (moveE(horse) | moveW(horse)) & validMoves;
        ourAttacks |= moves;
        numberOfMoves += popcount(moves[TOP]) + popcount(moves[MID]) +
                         popcount(moves[BOTTOM]);
      }
    }
  }

  // Dragon moves
  {
    pieces = ~pinnedPieces & board[BB::Type::ROOK] &
             board[BB::Type::ALL_WHITE] & board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      Bitboard dragon(square);
      moves = (LookUpTables::getRankAttacks(square, occupied) |
               LookUpTables::getFileAttacks(square, occupied) | moveNW(dragon) |
               moveNE(dragon) | moveSE(dragon) | moveSW(dragon)) &
              validMoves;
      ourAttacks |= moves;
      numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                        popcount(moves[BOTTOM]));
    }

    pieces = pinnedPieces & board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
             board[BB::Type::PROMOTED];
    if (pieces) {
      iterator.Init(pieces & kingRayRank);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        Bitboard dragon(square);
        moves = LookUpTables::getRankAttacks(square, occupied) & validMoves;
        ourAttacks |= moves;
        numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                          popcount(moves[BOTTOM]));
      }
      iterator.Init(pieces & kingRayFile);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        Bitboard dragon(square);
        moves = LookUpTables::getFileAttacks(square, occupied) & validMoves;
        ourAttacks |= moves;
        numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                          popcount(moves[BOTTOM]));
      }
      iterator.Init(pieces & kingRayDiagRight);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        Bitboard dragon(square);
        moves = (moveNE(dragon) | moveSW(dragon)) & validMoves;
        ourAttacks |= moves;
        numberOfMoves += popcount(moves[TOP]) + popcount(moves[MID]) +
                         popcount(moves[BOTTOM]);
      }
      iterator.Init(pieces & kingRayDiagLeft);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        Bitboard dragon(square);
        moves = (moveNW(dragon) | moveSE(dragon)) & validMoves;
        ourAttacks |= moves;
        numberOfMoves += popcount(moves[TOP]) + popcount(moves[MID]) +
                         popcount(moves[BOTTOM]);
      }
    }
  }

  // Drop moves
  {
    Bitboard legalDropSpots;
    // Pawns
    if (board.inHand.pieceNumber.WhitePawn > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last rank
      legalDropSpots[BOTTOM] &= ~BOTTOM_RANK;
      // Cannot drop to give checkmate
      pieces = board[BB::Type::KING] & board[BB::Type::ALL_BLACK];
      // All valid enemy king moves
      moves =
          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
          ~ourAttacks & ~board[BB::Type::ALL_BLACK];
      // If there is only one spot pawn cannot block it
      if (popcount(moves[TOP]) + popcount(moves[MID]) +
              popcount(moves[BOTTOM]) ==
          1) {
        legalDropSpots &= ~moveN(moves);
      }
      // Cannot drop on file with other pawn
      Bitboard validFiles;
      for (int fileIdx = 0; fileIdx < BOARD_DIM; fileIdx++) {
        Bitboard file = getFullFile(fileIdx);
        if (!(file & board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
              ~board[BB::Type::PROMOTED])) {
          validFiles |= file;
        }
      }
      legalDropSpots &= validFiles;
      numberOfMoves += popcount(legalDropSpots[TOP]) +
                       popcount(legalDropSpots[MID]) +
                       popcount(legalDropSpots[BOTTOM]);
    }
    if (board.inHand.pieceNumber.WhiteLance > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last rank
      legalDropSpots[BOTTOM] &= ~BOTTOM_RANK;
      numberOfMoves += popcount(legalDropSpots[TOP]) +
                       popcount(legalDropSpots[MID]) +
                       popcount(legalDropSpots[BOTTOM]);
    }
    if (board.inHand.pieceNumber.WhiteKnight > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last two ranks
      legalDropSpots[BOTTOM] &= TOP_RANK;
      numberOfMoves += popcount(legalDropSpots[TOP]) +
                       popcount(legalDropSpots[MID]) +
                       popcount(legalDropSpots[BOTTOM]);
    }
    legalDropSpots = validMoves & ~occupied;
    numberOfMoves +=
        ((board.inHand.pieceNumber.WhiteSilverGeneral > 0) +
         (board.inHand.pieceNumber.WhiteGoldGeneral > 0) +
         (board.inHand.pieceNumber.WhiteBishop > 0) +
         (board.inHand.pieceNumber.WhiteRook > 0)) *
        (popcount(legalDropSpots[TOP]) + popcount(legalDropSpots[MID]) +
         popcount(legalDropSpots[BOTTOM]));
  }

  return numberOfMoves;
}

RUNTYPE uint32_t countBlackMoves(const Board& board,
                                 Bitboard& pinned,
                                 Bitboard& validMoves,
                                 Bitboard& attackedByEnemy) {
  Bitboard pieces, moves, ourAttacks;
  Bitboard occupied = board[BB::Type::ALL_WHITE] | board[BB::Type::ALL_BLACK];
  BitboardIterator iterator;
  Square square;
  uint32_t numberOfMoves = 0;
  Bitboard pinnedPieces = pinned & board[BB::Type::ALL_BLACK];
  Bitboard pinningPieces = pinned & board[BB::Type::ALL_WHITE];
  Bitboard kingRayRank, kingRayFile, kingRayDiagRight, kingRayDiagLeft;

  // King Moves
  {
    pieces = board[BB::Type::KING] & board[BB::Type::ALL_BLACK];
    moves = moveNE(pieces) | moveN(pieces) | moveNW(pieces) | moveE(pieces) |
            moveW(pieces) | moveSE(pieces) | moveS(pieces) | moveSW(pieces);
    moves &= ~attackedByEnemy & ~board[BB::Type::ALL_BLACK];
    numberOfMoves +=
        popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);

    if (pinnedPieces) {
      iterator.Init(pieces);
      iterator.Next();
      Square kingSquare = iterator.GetCurrentSquare();
      Bitboard empty = {0, 0, 0};
      kingRayRank = LookUpTables::getRankAttacks(kingSquare, empty);
      kingRayFile = LookUpTables::getFileAttacks(kingSquare, empty);
      kingRayDiagRight = LookUpTables::getDiagRightAttacks(kingSquare, empty);
      kingRayDiagLeft = LookUpTables::getDiagLeftAttacks(kingSquare, empty);
    }
  }

  // Pawn moves
  {
    pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
             ~board[BB::Type::PROMOTED];
    moves = moveN((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayFile)) &
            validMoves;
    ourAttacks |= moves;
    numberOfMoves += popcount(moves[TOP] & TOP_RANK) +  // forced promotions
                     popcount(moves[TOP] & ~TOP_RANK) * 2 +  // promotions
                     popcount(moves[MID]) + popcount(moves[BOTTOM]);
  }

  // Knight moves
  {
    pieces = ~pinnedPieces & board[BB::Type::KNIGHT] &
             board[BB::Type::ALL_BLACK] & ~board[BB::Type::PROMOTED];
    moves = moveN(moveNE(pieces)) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += popcount(moves[TOP] & ~BOTTOM_RANK) +  // forced promotions
                     popcount(moves[TOP] & BOTTOM_RANK) * 2 +  // promotions
                     popcount(moves[MID]) + popcount(moves[BOTTOM]);
    moves = moveN(moveNW(pieces)) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += popcount(moves[TOP] & ~BOTTOM_RANK) +  // forced promotions
                     popcount(moves[TOP] & BOTTOM_RANK) * 2 +  // promotions
                     popcount(moves[MID]) + popcount(moves[BOTTOM]);
  }

  // SilverGenerals moves
  {
    pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_BLACK] &
             ~board[BB::Type::PROMOTED];
    moves = moveN((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayFile)) &
            validMoves;
    ourAttacks |= moves;
    numberOfMoves += popcount(moves[TOP]) * 2 +  // promotions
                     popcount(moves[MID]) + popcount(moves[BOTTOM]);
    moves = moveNE((pieces & ~pinnedPieces) |
                   (pieces & pinnedPieces & kingRayDiagRight)) &
            validMoves;
    ourAttacks |= moves;
    numberOfMoves += popcount(moves[TOP]) * 2 +  // promotions
                     popcount(moves[MID]) + popcount(moves[BOTTOM]);
    moves = moveNW((pieces & ~pinnedPieces) |
                   (pieces & pinnedPieces & kingRayDiagLeft)) &
            validMoves;
    ourAttacks |= moves;
    numberOfMoves += popcount(moves[TOP]) * 2 +  // promotions
                     popcount(moves[MID]) + popcount(moves[BOTTOM]);
    moves = moveSE((pieces & ~pinnedPieces) |
                   (pieces & pinnedPieces & kingRayDiagLeft)) &
            validMoves;
    ourAttacks |= moves;
    numberOfMoves += popcount(moves[TOP]) * 2 +  // promotions
                     popcount(moves[MID] & TOP_RANK) *
                         2 +  // promotion when starting from promotion zone
                     popcount(moves[MID] & ~TOP_RANK) +
                     popcount(moves[BOTTOM]);
    moves = moveSW((pieces & ~pinnedPieces) |
                   (pieces & pinnedPieces & kingRayDiagRight)) &
            validMoves;
    ourAttacks |= moves;
    numberOfMoves += popcount(moves[TOP]) * 2 +  // promotions
                     popcount(moves[MID] & TOP_RANK) *
                         2 +  // promotion when starting from promotion zone
                     popcount(moves[MID] & ~TOP_RANK) +
                     popcount(moves[BOTTOM]);
  }

  // GoldGenerals moves
  {
    pieces = (board[BB::Type::GOLD_GENERAL] |
              ((board[BB::Type::PAWN] | board[BB::Type::LANCE] |
                board[BB::Type::KNIGHT] | board[BB::Type::SILVER_GENERAL]) &
               board[BB::Type::PROMOTED])) &
             board[BB::Type::ALL_BLACK];
    moves = moveN((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayFile)) &
            validMoves;
    ourAttacks |= moves;
    numberOfMoves +=
        popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
    moves = moveNE((pieces & ~pinnedPieces) |
                   (pieces & pinnedPieces & kingRayDiagRight)) &
            validMoves;
    ourAttacks |= moves;
    numberOfMoves +=
        popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
    moves = moveNW((pieces & ~pinnedPieces) |
                   (pieces & pinnedPieces & kingRayDiagLeft)) &
            validMoves;
    ourAttacks |= moves;
    numberOfMoves +=
        popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
    moves = moveE((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayRank)) &
            validMoves;
    ourAttacks |= moves;
    numberOfMoves +=
        popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
    moves = moveW((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayRank)) &
            validMoves;
    ourAttacks |= moves;
    numberOfMoves +=
        popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
    moves = moveS((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayFile)) &
            validMoves;
    ourAttacks |= moves;
    numberOfMoves +=
        popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
  }

  // Lance moves
  {
    pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] &
             ~board[BB::Type::PROMOTED];
    iterator.Init((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayFile));
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      moves = LookUpTables::getFileAttacks(square, occupied) &
              LookUpTables::getRankMask(squareToRank(square)) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += popcount(moves[TOP] & TOP_RANK) +  // forced promotions
                       popcount(moves[TOP] & ~TOP_RANK) * 2 +  // promotions
                       popcount(moves[MID]) + popcount(moves[BOTTOM]);
    }
  }

  // Bishop moves
  {
    pieces = ~pinnedPieces & board[BB::Type::BISHOP] &
             board[BB::Type::ALL_BLACK] & ~board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      moves = (LookUpTables::getDiagRightAttacks(square, occupied) |
               LookUpTables::getDiagLeftAttacks(square, occupied)) &
              validMoves;
      ourAttacks |= moves;
      if (square <= BLACK_PROMOTION_END) {  // Starting from promotion zone
        numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                          popcount(moves[BOTTOM])) *
                         2;
      } else {
        numberOfMoves += popcount(moves[TOP]) * 2 +  // end in promotion Zone
                         popcount(moves[MID]) + popcount(moves[BOTTOM]);
      }
    }

    pieces = pinnedPieces & board[BB::Type::BISHOP] &
             board[BB::Type::ALL_BLACK] & ~board[BB::Type::PROMOTED];
    if (pieces) {
      iterator.Init(pieces & kingRayDiagLeft);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        moves = LookUpTables::getDiagLeftAttacks(square, occupied) & validMoves;
        ourAttacks |= moves;
        if (square <= BLACK_PROMOTION_END) {  // Starting from promotion zone
          numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                            popcount(moves[BOTTOM])) *
                           2;
        } else {
          numberOfMoves += popcount(moves[TOP]) * 2 +  // end in promotion Zone
                           popcount(moves[MID]) + popcount(moves[BOTTOM]);
        }
      }
      iterator.Init(pieces & kingRayDiagRight);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        moves =
            LookUpTables::getDiagRightAttacks(square, occupied) & validMoves;
        ourAttacks |= moves;
        if (square <= BLACK_PROMOTION_END) {  // Starting from promotion zone
          numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                            popcount(moves[BOTTOM])) *
                           2;
        } else {
          numberOfMoves += popcount(moves[TOP]) * 2 +  // end in promotion Zone
                           popcount(moves[MID]) + popcount(moves[BOTTOM]);
        }
      }
    }
  }

  // Rook moves
  {
    pieces = ~pinnedPieces & board[BB::Type::ROOK] &
             board[BB::Type::ALL_BLACK] & ~board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      moves = (LookUpTables::getRankAttacks(square, occupied) |
               LookUpTables::getFileAttacks(square, occupied)) &
              validMoves;
      ourAttacks |= moves;
      if (square <= BLACK_PROMOTION_END) {  // Starting from promotion zone
        numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                          popcount(moves[BOTTOM])) *
                         2;
      } else {
        numberOfMoves += popcount(moves[TOP]) * 2 +  // end in promotion Zone
                         popcount(moves[MID]) + popcount(moves[BOTTOM]);
      }
    }

    pieces = pinnedPieces & board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
             ~board[BB::Type::PROMOTED];
    if (pieces) {
      iterator.Init(pieces & kingRayRank);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        moves = LookUpTables::getRankAttacks(square, occupied) & validMoves;
        ourAttacks |= moves;
        if (square <= BLACK_PROMOTION_END) {  // Starting from promotion zone
          numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                            popcount(moves[BOTTOM])) *
                           2;
        } else {
          numberOfMoves += popcount(moves[TOP]) * 2 +  // end in promotion Zone
                           popcount(moves[MID]) + popcount(moves[BOTTOM]);
        }
      }
      iterator.Init(pieces & kingRayFile);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        moves = LookUpTables::getFileAttacks(square, occupied) & validMoves;
        ourAttacks |= moves;
        if (square <= BLACK_PROMOTION_END) {  // Starting from promotion zone
          numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                            popcount(moves[BOTTOM])) *
                           2;
        } else {
          numberOfMoves += popcount(moves[TOP]) * 2 +  // end in promotion Zone
                           popcount(moves[MID]) + popcount(moves[BOTTOM]);
        }
      }
    }
  }

  // Horse moves
  {
    pieces = ~pinnedPieces & board[BB::Type::BISHOP] &
             board[BB::Type::ALL_BLACK] & board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      Bitboard horse = Bitboard(square);
      moves = (LookUpTables::getDiagRightAttacks(square, occupied) |
               LookUpTables::getDiagLeftAttacks(square, occupied) |
               moveN(horse) | moveE(horse) | moveS(horse) | moveW(horse)) &
              validMoves;
      ourAttacks |= moves;
      numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                        popcount(moves[BOTTOM]));
    }

    pieces = pinnedPieces & board[BB::Type::BISHOP] &
             board[BB::Type::ALL_BLACK] & board[BB::Type::PROMOTED];
    if (pieces) {
      iterator.Init(pieces & kingRayDiagLeft);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        moves = LookUpTables::getDiagLeftAttacks(square, occupied) & validMoves;
        ourAttacks |= moves;
        numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                          popcount(moves[BOTTOM]));
      }
      iterator.Init(pieces & kingRayDiagRight);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        moves =
            LookUpTables::getDiagRightAttacks(square, occupied) & validMoves;
        ourAttacks |= moves;
        numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                          popcount(moves[BOTTOM]));
      }
      iterator.Init(pieces & kingRayFile);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        Bitboard horse(square);
        moves = (moveN(horse) | moveS(horse)) & validMoves;
        ourAttacks |= moves;
        numberOfMoves += popcount(moves[TOP]) + popcount(moves[MID]) +
                         popcount(moves[BOTTOM]);
      }
      iterator.Init(pieces & kingRayRank);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        Bitboard horse(square);
        moves = (moveE(horse) | moveW(horse)) & validMoves;
        ourAttacks |= moves;
        numberOfMoves += popcount(moves[TOP]) + popcount(moves[MID]) +
                         popcount(moves[BOTTOM]);
      }
    }
  }

  // Dragon moves
  {
    pieces = ~pinnedPieces & board[BB::Type::ROOK] &
             board[BB::Type::ALL_BLACK] & board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.Next()) {
      square = iterator.GetCurrentSquare();
      Bitboard dragon(square);
      moves = (LookUpTables::getRankAttacks(square, occupied) |
               LookUpTables::getFileAttacks(square, occupied) | moveNW(dragon) |
               moveNE(dragon) | moveSE(dragon) | moveSW(dragon)) &
              validMoves;
      ourAttacks |= moves;
      numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                        popcount(moves[BOTTOM]));
    }

    pieces = pinnedPieces & board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
             board[BB::Type::PROMOTED];
    if (pieces) {
      iterator.Init(pieces & kingRayRank);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        Bitboard dragon(square);
        moves = LookUpTables::getRankAttacks(square, occupied) & validMoves;
        ourAttacks |= moves;
        numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                          popcount(moves[BOTTOM]));
      }
      iterator.Init(pieces & kingRayFile);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        Bitboard dragon(square);
        moves = LookUpTables::getFileAttacks(square, occupied) & validMoves;
        ourAttacks |= moves;
        numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                          popcount(moves[BOTTOM]));
      }
      iterator.Init(pieces & kingRayDiagRight);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        Bitboard dragon(square);
        moves = (moveNE(dragon) | moveSW(dragon)) & validMoves;
        ourAttacks |= moves;
        numberOfMoves += popcount(moves[TOP]) + popcount(moves[MID]) +
                         popcount(moves[BOTTOM]);
      }
      iterator.Init(pieces & kingRayDiagLeft);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        Bitboard dragon(square);
        moves = (moveNW(dragon) | moveSE(dragon)) & validMoves;
        ourAttacks |= moves;
        numberOfMoves += popcount(moves[TOP]) + popcount(moves[MID]) +
                         popcount(moves[BOTTOM]);
      }
    }
  }

  // Drop moves
  {
    Bitboard legalDropSpots;
    // Pawns
    if (board.inHand.pieceNumber.BlackPawn > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last rank
      legalDropSpots[TOP] &= ~TOP_RANK;
      // Cannot drop to give checkmate
      pieces = board[BB::Type::KING] & board[BB::Type::ALL_WHITE];
      // All valid enemy king moves
      moves =
          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
          ~ourAttacks & ~board[BB::Type::ALL_WHITE];
      // If there is only one spot pawn cannot block it
      if (popcount(moves[TOP]) + popcount(moves[MID]) +
              popcount(moves[BOTTOM]) ==
          1) {
        legalDropSpots &= ~moveS(moves);
      }
      // Cannot drop on file with other pawn
      Bitboard validFiles;
      for (int fileIdx = 0; fileIdx < BOARD_DIM; fileIdx++) {
        Bitboard file = getFullFile(fileIdx);
        if (!(file & board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
              ~board[BB::Type::PROMOTED])) {
          validFiles |= file;
        }
      }
      legalDropSpots &= validFiles;
      numberOfMoves += popcount(legalDropSpots[TOP]) +
                       popcount(legalDropSpots[MID]) +
                       popcount(legalDropSpots[BOTTOM]);
    }
    if (board.inHand.pieceNumber.BlackLance > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last rank
      legalDropSpots[TOP] &= ~TOP_RANK;
      numberOfMoves += popcount(legalDropSpots[TOP]) +
                       popcount(legalDropSpots[MID]) +
                       popcount(legalDropSpots[BOTTOM]);
    }
    if (board.inHand.pieceNumber.BlackKnight > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last two ranks
      legalDropSpots[TOP] &= BOTTOM_RANK;
      numberOfMoves += popcount(legalDropSpots[TOP]) +
                       popcount(legalDropSpots[MID]) +
                       popcount(legalDropSpots[BOTTOM]);
    }
    legalDropSpots = validMoves & ~occupied;
    numberOfMoves +=
        ((board.inHand.pieceNumber.BlackSilverGeneral > 0) +
         (board.inHand.pieceNumber.BlackGoldGeneral > 0) +
         (board.inHand.pieceNumber.BlackBishop > 0) +
         (board.inHand.pieceNumber.BlackRook > 0)) *
        (popcount(legalDropSpots[TOP]) + popcount(legalDropSpots[MID]) +
         popcount(legalDropSpots[BOTTOM]));
  }
  return numberOfMoves;
}

RUNTYPE uint32_t generateWhiteMoves(const Board& board,
                                    Bitboard& pinned,
                                    Bitboard& validMoves,
                                    Bitboard& attackedByEnemy,
                                    Move* movesArray) {
  Bitboard pieces, moves, ourAttacks;
  Bitboard occupied = board[BB::Type::ALL_WHITE] | board[BB::Type::ALL_BLACK];
  BitboardIterator movesIterator, iterator;
  Move move;
  uint32_t moveNumber = 0;
  Bitboard pinnedPieces = pinned & board[BB::Type::ALL_WHITE];
  Bitboard pinningPieces = pinned & board[BB::Type::ALL_BLACK];
  Bitboard kingRayRank, kingRayFile, kingRayDiagRight, kingRayDiagLeft;

  // King moves
  move.promotion = 0;
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
        movesArray[moveNumber] = move;
        moveNumber++;
      }

      if (pinnedPieces) {
        Square kingSquare = static_cast<Square>(move.from);
        Bitboard empty = {0, 0, 0};
        kingRayRank = LookUpTables::getRankAttacks(kingSquare, empty);
        kingRayFile = LookUpTables::getFileAttacks(kingSquare, empty);
        kingRayDiagRight = LookUpTables::getDiagRightAttacks(kingSquare, empty);
        kingRayDiagLeft = LookUpTables::getDiagLeftAttacks(kingSquare, empty);
      }
    }
  }

  // Pawn moves
  {
    pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
             ~board[BB::Type::PROMOTED];
    moves = moveS((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayFile)) &
            validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + N;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        movesArray[moveNumber] = move;
        moveNumber++;
      }
      // Not when forced promotion
      if (move.to < WHITE_PAWN_LANCE_FORECED_PROMOTION_START) {
        move.promotion = 0;
        movesArray[moveNumber] = move;
        moveNumber++;
      }
    }
  }

  // Knight moves
  {
    pieces = ~pinnedPieces & board[BB::Type::KNIGHT] &
             board[BB::Type::ALL_WHITE] & ~board[BB::Type::PROMOTED];
    moves = moveS(moveSE(pieces)) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + N + NW;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        movesArray[moveNumber] = move;
        moveNumber++;
      }
      // Not when forced promotion
      if (move.to < WHITE_HORSE_FORCED_PROMOTION_START) {
        move.promotion = 0;
        movesArray[moveNumber] = move;
        moveNumber++;
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
        movesArray[moveNumber] = move;
        moveNumber++;
      }
      // Not when forced promotion
      if (move.to < WHITE_HORSE_FORCED_PROMOTION_START) {
        move.promotion = 0;
        movesArray[moveNumber] = move;
        moveNumber++;
      }
    }
  }

  // SilverGenerals moves
  {
    pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_WHITE] &
             ~board[BB::Type::PROMOTED];
    moves = moveS((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayFile)) &
            validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + N;
      move.promotion = 0;
      movesArray[moveNumber] = move;
      moveNumber++;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        movesArray[moveNumber] = move;
        moveNumber++;
      }
    }
    moves = moveSE((pieces & ~pinnedPieces) |
                   (pieces & pinnedPieces & kingRayDiagLeft)) &
            validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + NW;
      move.promotion = 0;
      movesArray[moveNumber] = move;
      moveNumber++;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        movesArray[moveNumber] = move;
        moveNumber++;
      }
    }
    moves = moveSW((pieces & ~pinnedPieces) |
                   (pieces & pinnedPieces & kingRayDiagRight)) &
            validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + NE;
      move.promotion = 0;
      movesArray[moveNumber] = move;
      moveNumber++;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        movesArray[moveNumber] = move;
        moveNumber++;
      }
    }
    moves = moveNE((pieces & ~pinnedPieces) |
                   (pieces & pinnedPieces & kingRayDiagRight)) &
            validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + SW;
      move.promotion = 0;
      movesArray[moveNumber] = move;
      moveNumber++;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START ||
          move.from >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        movesArray[moveNumber] = move;
        moveNumber++;
      }
    }
    moves = moveNW((pieces & ~pinnedPieces) |
                   (pieces & pinnedPieces & kingRayDiagLeft)) &
            validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + SE;
      move.promotion = 0;
      movesArray[moveNumber] = move;
      moveNumber++;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START ||
          move.from >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        movesArray[moveNumber] = move;
        moveNumber++;
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
    moves = moveS((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayFile)) &
            validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + N;
      move.promotion = 0;
      movesArray[moveNumber] = move;
      moveNumber++;
    }
    moves = moveSE((pieces & ~pinnedPieces) |
                   (pieces & pinnedPieces & kingRayDiagLeft)) &
            validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + NW;
      move.promotion = 0;
      movesArray[moveNumber] = move;
      moveNumber++;
    }
    moves = moveSW((pieces & ~pinnedPieces) |
                   (pieces & pinnedPieces & kingRayDiagRight)) &
            validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + NE;
      move.promotion = 0;
      movesArray[moveNumber] = move;
      moveNumber++;
    }
    moves = moveE((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayRank)) &
            validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + W;
      move.promotion = 0;
      movesArray[moveNumber] = move;
      moveNumber++;
    }
    moves = moveW((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayRank)) &
            validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + E;
      move.promotion = 0;
      movesArray[moveNumber] = move;
      moveNumber++;
    }
    moves = moveN((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayFile)) &
            validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + S;
      move.promotion = 0;
      movesArray[moveNumber] = move;
      moveNumber++;
    }
  }

  // Lances moves
  {
    pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] &
             ~board[BB::Type::PROMOTED];
    iterator.Init((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayFile));
    while (iterator.Next()) {
      move.from = iterator.GetCurrentSquare();
      moves = LookUpTables::getFileAttacks(static_cast<Square>(move.from),
                                           occupied) &
              ~LookUpTables::getRankMask(
                  squareToRank(static_cast<Square>(move.from))) &
              validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        // Promotion
        if (move.to >= WHITE_PROMOTION_START) {
          move.promotion = 1;
          movesArray[moveNumber] = move;
          moveNumber++;
        }
        // Not when forced promotion
        if (move.to < WHITE_PAWN_LANCE_FORECED_PROMOTION_START) {
          move.promotion = 0;
          movesArray[moveNumber] = move;
          moveNumber++;
        }
      }
    }
  }

  // Bishop moves
  {
    pieces = ~pinnedPieces & board[BB::Type::BISHOP] &
             board[BB::Type::ALL_WHITE] & ~board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.Next()) {
      move.from = iterator.GetCurrentSquare();
      moves = (LookUpTables::getDiagRightAttacks(static_cast<Square>(move.from),
                                                 occupied) |
               LookUpTables::getDiagLeftAttacks(static_cast<Square>(move.from),
                                                occupied)) &
              validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.promotion = 0;
        movesArray[moveNumber] = move;
        moveNumber++;
        // Promotion
        if (move.to >= WHITE_PROMOTION_START ||
            move.from >= WHITE_PROMOTION_START) {
          move.promotion = 1;
          movesArray[moveNumber] = move;
          moveNumber++;
        }
      }
    }

    pieces = pinnedPieces & board[BB::Type::BISHOP] &
             board[BB::Type::ALL_WHITE] & ~board[BB::Type::PROMOTED];
    if (pieces) {
      iterator.Init(pieces & kingRayDiagLeft);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        moves = LookUpTables::getDiagLeftAttacks(static_cast<Square>(move.from),
                                                 occupied) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          movesArray[moveNumber] = move;
          moveNumber++;
          // Promotion
          if (move.to >= WHITE_PROMOTION_START ||
              move.from >= WHITE_PROMOTION_START) {
            move.promotion = 1;
            movesArray[moveNumber] = move;
            moveNumber++;
          }
        }
      }
      iterator.Init(pieces & kingRayDiagRight);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        moves = LookUpTables::getDiagRightAttacks(
                    static_cast<Square>(move.from), occupied) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          movesArray[moveNumber] = move;
          moveNumber++;
          // Promotion
          if (move.to >= WHITE_PROMOTION_START ||
              move.from >= WHITE_PROMOTION_START) {
            move.promotion = 1;
            movesArray[moveNumber] = move;
            moveNumber++;
          }
        }
      }
    }
  }

  // Rook moves
  {
    pieces = ~pinnedPieces & board[BB::Type::ROOK] &
             board[BB::Type::ALL_WHITE] & ~board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.Next()) {
      move.from = iterator.GetCurrentSquare();
      moves = (LookUpTables::getRankAttacks(static_cast<Square>(move.from),
                                            occupied) |
               LookUpTables::getFileAttacks(static_cast<Square>(move.from),
                                            occupied)) &
              validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.promotion = 0;
        movesArray[moveNumber] = move;
        moveNumber++;
        // Promotion
        if (move.to >= WHITE_PROMOTION_START ||
            move.from >= WHITE_PROMOTION_START) {
          move.promotion = 1;
          movesArray[moveNumber] = move;
          moveNumber++;
        }
      }
    }

    pieces = pinnedPieces & board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
             ~board[BB::Type::PROMOTED];
    if (pieces) {
      iterator.Init(pieces & kingRayFile);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        moves = LookUpTables::getFileAttacks(static_cast<Square>(move.from),
                                             occupied) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          movesArray[moveNumber] = move;
          moveNumber++;
          // Promotion
          if (move.to >= WHITE_PROMOTION_START ||
              move.from >= WHITE_PROMOTION_START) {
            move.promotion = 1;
            movesArray[moveNumber] = move;
            moveNumber++;
          }
        }
      }
      iterator.Init(pieces & kingRayRank);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        moves = LookUpTables::getRankAttacks(static_cast<Square>(move.from),
                                             occupied) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          movesArray[moveNumber] = move;
          moveNumber++;
          // Promotion
          if (move.to >= WHITE_PROMOTION_START ||
              move.from >= WHITE_PROMOTION_START) {
            move.promotion = 1;
            movesArray[moveNumber] = move;
            moveNumber++;
          }
        }
      }
    }
  }

  move.promotion = 0;
  // Horse moves
  {
    pieces = ~pinnedPieces & board[BB::Type::BISHOP] &
             board[BB::Type::ALL_WHITE] & board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.Next()) {
      move.from = iterator.GetCurrentSquare();
      Bitboard horse(static_cast<Square>(move.from));
      moves = (LookUpTables::getDiagRightAttacks(static_cast<Square>(move.from),
                                                 occupied) |
               LookUpTables::getDiagLeftAttacks(static_cast<Square>(move.from),
                                                occupied) |
               moveN(horse) | moveE(horse) | moveS(horse) | moveW(horse)) &
              validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        movesArray[moveNumber] = move;
        moveNumber++;
      }
    }

    pieces = pinnedPieces & board[BB::Type::BISHOP] &
             board[BB::Type::ALL_WHITE] & board[BB::Type::PROMOTED];
    if (pieces) {
      iterator.Init(pieces & kingRayDiagLeft);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        moves = LookUpTables::getDiagLeftAttacks(static_cast<Square>(move.from),
                                                 occupied) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          movesArray[moveNumber] = move;
          moveNumber++;
        }
      }
      iterator.Init(pieces & kingRayDiagRight);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        moves = LookUpTables::getDiagRightAttacks(
                    static_cast<Square>(move.from), occupied) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          movesArray[moveNumber] = move;
          moveNumber++;
        }
      }
      iterator.Init(pieces & kingRayFile);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        Bitboard horse(static_cast<Square>(move.from));
        moves = (moveN(horse) | moveS(horse)) & validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          movesArray[moveNumber] = move;
          moveNumber++;
        }
      }
      iterator.Init(pieces & kingRayRank);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        Bitboard horse(static_cast<Square>(move.from));
        moves = (moveE(horse) | moveW(horse)) & validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          movesArray[moveNumber] = move;
          moveNumber++;
        }
      }
    }
  }

  // Dragon moves
  {
    pieces = ~pinnedPieces & board[BB::Type::ROOK] &
             board[BB::Type::ALL_WHITE] & board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.Next()) {
      move.from = iterator.GetCurrentSquare();
      Bitboard dragon(static_cast<Square>(move.from));
      moves =
          (LookUpTables::getRankAttacks(static_cast<Square>(move.from),
                                        occupied) |
           LookUpTables::getFileAttacks(static_cast<Square>(move.from),
                                        occupied) |
           moveNW(pieces) | moveNE(pieces) | moveSE(pieces) | moveSW(pieces)) &
          validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        movesArray[moveNumber] = move;
        moveNumber++;
      }
    }

    pieces = pinnedPieces & board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
             board[BB::Type::PROMOTED];
    if (pieces) {
      iterator.Init(pieces & kingRayFile);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        moves = LookUpTables::getFileAttacks(static_cast<Square>(move.from),
                                             occupied) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          movesArray[moveNumber] = move;
          moveNumber++;
        }
      }
      iterator.Init(pieces & kingRayRank);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        moves = LookUpTables::getRankAttacks(static_cast<Square>(move.from),
                                             occupied) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          movesArray[moveNumber] = move;
          moveNumber++;
        }
      }
      iterator.Init(pieces & kingRayDiagRight);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        Bitboard dragon(static_cast<Square>(move.from));
        moves = (moveNE(dragon) | moveSW(dragon)) & validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          movesArray[moveNumber] = move;
          moveNumber++;
        }
      }
      iterator.Init(pieces & kingRayDiagLeft);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        Bitboard dragon(static_cast<Square>(move.from));
        moves = (moveNW(dragon) | moveSE(dragon)) & validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          movesArray[moveNumber] = move;
          moveNumber++;
        }
      }
    }
  }

  // Generate Drop moves
  {
    Bitboard legalDropSpots;
    // Pawns
    if (board.inHand.pieceNumber.WhitePawn > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last rank
      legalDropSpots[BOTTOM] &= ~BOTTOM_RANK;
      // Cannot drop to give checkmate
      pieces = board[BB::Type::KING] & board[BB::Type::ALL_BLACK];
      // All valid enemy king moves
      moves =
          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
          ~ourAttacks & ~board[BB::Type::ALL_BLACK];
      // If there is only one spot pawn cannot block it
      if (popcount(moves[TOP]) + popcount(moves[MID]) +
              popcount(moves[BOTTOM]) ==
          1) {
        legalDropSpots &= ~moveN(moves);
      }
      // Cannot drop on file with other pawn
      Bitboard validFiles;
      for (int fileIdx = 0; fileIdx < BOARD_DIM; fileIdx++) {
        Bitboard file = getFullFile(fileIdx);
        if (!(file & board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
              ~board[BB::Type::PROMOTED])) {
          validFiles |= file;
        }
      }
      legalDropSpots &= validFiles;
      movesIterator.Init(legalDropSpots);
      move.from = WHITE_PAWN_DROP;
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        movesArray[moveNumber] = move;
        moveNumber++;
      }
    }
    if (board.inHand.pieceNumber.WhiteLance > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last rank
      legalDropSpots[BOTTOM] &= ~BOTTOM_RANK;
      movesIterator.Init(legalDropSpots);
      move.from = WHITE_LANCE_DROP;
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        movesArray[moveNumber] = move;
        moveNumber++;
      }
    }
    if (board.inHand.pieceNumber.WhiteKnight > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last two ranks
      legalDropSpots[BOTTOM] &= TOP_RANK;
      movesIterator.Init(legalDropSpots);
      move.from = WHITE_KNIGHT_DROP;
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        movesArray[moveNumber] = move;
        moveNumber++;
      }
    }
    legalDropSpots = validMoves & ~occupied;
    movesIterator.Init(legalDropSpots);
    while (movesIterator.Next()) {
      if (board.inHand.pieceNumber.WhiteSilverGeneral > 0) {
        move.from = WHITE_SILVER_GENERAL_DROP;
        move.to = movesIterator.GetCurrentSquare();
        movesArray[moveNumber] = move;
        moveNumber++;
      }
      if (board.inHand.pieceNumber.WhiteGoldGeneral > 0) {
        move.from = WHITE_GOLD_GENERAL_DROP;
        move.to = movesIterator.GetCurrentSquare();
        movesArray[moveNumber] = move;
        moveNumber++;
      }
      if (board.inHand.pieceNumber.WhiteBishop > 0) {
        move.from = WHITE_BISHOP_DROP;
        move.to = movesIterator.GetCurrentSquare();
        movesArray[moveNumber] = move;
        moveNumber++;
      }
      if (board.inHand.pieceNumber.WhiteRook > 0) {
        move.from = WHITE_ROOK_DROP;
        move.to = movesIterator.GetCurrentSquare();
        movesArray[moveNumber] = move;
        moveNumber++;
      }
    }
  }

  return moveNumber;
}

RUNTYPE uint32_t generateBlackMoves(const Board& board,
                                    Bitboard& pinned,
                                    Bitboard& validMoves,
                                    Bitboard& attackedByEnemy,
                                    Move* movesArray) {
  Bitboard pieces, moves, ourAttacks;
  Bitboard occupied = board[BB::Type::ALL_WHITE] | board[BB::Type::ALL_BLACK];
  BitboardIterator movesIterator, iterator;
  Move move;
  uint32_t moveNumber = 0;
  Bitboard pinnedPieces = pinned & board[BB::Type::ALL_BLACK];
  Bitboard pinningPieces = pinned & board[BB::Type::ALL_WHITE];
  Bitboard kingRayRank, kingRayFile, kingRayDiagRight, kingRayDiagLeft;

  // King moves
  move.promotion = 0;
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
        movesArray[moveNumber] = move;
        moveNumber++;
      }
    }

    if (pinnedPieces) {
      iterator.Init(pieces);
      iterator.Next();
      Square kingSquare = iterator.GetCurrentSquare();
      Bitboard empty = {0, 0, 0};
      kingRayRank = LookUpTables::getRankAttacks(kingSquare, empty);
      kingRayFile = LookUpTables::getFileAttacks(kingSquare, empty);
      kingRayDiagRight = LookUpTables::getDiagRightAttacks(kingSquare, empty);
      kingRayDiagLeft = LookUpTables::getDiagLeftAttacks(kingSquare, empty);
    }
  }

  // Pawn moves
  {
    pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
             ~board[BB::Type::PROMOTED];
    moves = moveN((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayFile)) &
            validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + S;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        movesArray[moveNumber] = move;
        moveNumber++;
      }
      // Not when forced promotion
      if (move.to > BLACK_PAWN_LANCE_FORCE_PROMOTION_END) {
        move.promotion = 0;
        movesArray[moveNumber] = move;
        moveNumber++;
      }
    }
  }

  // Knight moves
  {
    pieces = ~pinnedPieces & board[BB::Type::KNIGHT] &
             board[BB::Type::ALL_BLACK] & ~board[BB::Type::PROMOTED];
    moves = moveN(moveNE(pieces)) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + S + SW;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        movesArray[moveNumber] = move;
        moveNumber++;
      }
      // Not when forced promotion
      if (move.to > BLACK_HORSE_FORCED_PROMOTION_END) {
        move.promotion = 0;
        movesArray[moveNumber] = move;
        moveNumber++;
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
        movesArray[moveNumber] = move;
        moveNumber++;
      }
      // Not when forced promotion
      if (move.to > BLACK_HORSE_FORCED_PROMOTION_END) {
        move.promotion = 0;
        movesArray[moveNumber] = move;
        moveNumber++;
      }
    }
  }

  // SilverGenerals moves
  {
    pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_BLACK] &
             ~board[BB::Type::PROMOTED];
    moves = moveN((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayFile)) &
            validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + S;
      move.promotion = 0;
      movesArray[moveNumber] = move;
      moveNumber++;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        movesArray[moveNumber] = move;
        moveNumber++;
      }
    }
    moves = moveNE((pieces & ~pinnedPieces) |
                   (pieces & pinnedPieces & kingRayDiagRight)) &
            validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + SW;
      move.promotion = 0;
      movesArray[moveNumber] = move;
      moveNumber++;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        movesArray[moveNumber] = move;
        moveNumber++;
      }
    }
    moves = moveNW((pieces & ~pinnedPieces) |
                   (pieces & pinnedPieces & kingRayDiagLeft)) &
            validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + SE;
      move.promotion = 0;
      movesArray[moveNumber] = move;
      moveNumber++;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        movesArray[moveNumber] = move;
        moveNumber++;
      }
    }
    moves = moveSE((pieces & ~pinnedPieces) |
                   (pieces & pinnedPieces & kingRayDiagLeft)) &
            validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + NW;
      move.promotion = 0;
      movesArray[moveNumber] = move;
      moveNumber++;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END || move.from <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        movesArray[moveNumber] = move;
        moveNumber++;
      }
    }
    moves = moveSW((pieces & ~pinnedPieces) |
                   (pieces & pinnedPieces & kingRayDiagRight)) &
            validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + NE;
      move.promotion = 0;
      movesArray[moveNumber] = move;
      moveNumber++;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END || move.from <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        movesArray[moveNumber] = move;
        moveNumber++;
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
    moves = moveN((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayFile)) &
            validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + S;
      move.promotion = 0;
      movesArray[moveNumber] = move;
      moveNumber++;
    }
    moves = moveNE((pieces & ~pinnedPieces) |
                   (pieces & pinnedPieces & kingRayDiagRight)) &
            validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + SW;
      move.promotion = 0;
      movesArray[moveNumber] = move;
      moveNumber++;
    }
    moves = moveNW((pieces & ~pinnedPieces) |
                   (pieces & pinnedPieces & kingRayDiagLeft)) &
            validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + SE;
      move.promotion = 0;
      movesArray[moveNumber] = move;
      moveNumber++;
    }
    moves = moveE((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayRank)) &
            validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + W;
      move.promotion = 0;
      movesArray[moveNumber] = move;
      moveNumber++;
    }
    moves = moveW((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayRank)) &
            validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + E;
      move.promotion = 0;
      movesArray[moveNumber] = move;
      moveNumber++;
    }
    moves = moveS((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayFile)) &
            validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + N;
      move.promotion = 0;
      movesArray[moveNumber] = move;
      moveNumber++;
    }
  }

  // Lances moves
  {
    pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] &
             ~board[BB::Type::PROMOTED];
    iterator.Init((pieces & ~pinnedPieces) |
                  (pieces & pinnedPieces & kingRayFile));
    while (iterator.Next()) {
      move.from = iterator.GetCurrentSquare();
      moves = LookUpTables::getFileAttacks(static_cast<Square>(move.from),
                                           occupied) &
              LookUpTables::getRankMask(
                  squareToRank(static_cast<Square>(move.from))) &
              validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        // Promotion
        if (move.to <= BLACK_PROMOTION_END) {
          move.promotion = 1;
          movesArray[moveNumber] = move;
          moveNumber++;
        }
        // Not when forced promotion
        if (move.to > BLACK_PAWN_LANCE_FORCE_PROMOTION_END) {
          move.promotion = 0;
          movesArray[moveNumber] = move;
          moveNumber++;
        }
      }
    }
  }

  // Bishop moves
  {
    pieces = ~pinnedPieces & board[BB::Type::BISHOP] &
             board[BB::Type::ALL_BLACK] & ~board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.Next()) {
      move.from = iterator.GetCurrentSquare();
      moves = (LookUpTables::getDiagRightAttacks(static_cast<Square>(move.from),
                                                 occupied) |
               LookUpTables::getDiagLeftAttacks(static_cast<Square>(move.from),
                                                occupied)) &
              validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.promotion = 0;
        movesArray[moveNumber] = move;
        moveNumber++;
        // Promotion
        if (move.to <= BLACK_PROMOTION_END ||
            move.from <= BLACK_PROMOTION_END) {
          move.promotion = 1;
          movesArray[moveNumber] = move;
          moveNumber++;
        }
      }
    }

    pieces = pinnedPieces & board[BB::Type::BISHOP] &
             board[BB::Type::ALL_BLACK] & ~board[BB::Type::PROMOTED];
    if (pieces) {
      iterator.Init(pieces & kingRayDiagLeft);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        moves = LookUpTables::getDiagLeftAttacks(static_cast<Square>(move.from),
                                                 occupied) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          movesArray[moveNumber] = move;
          moveNumber++;
          // Promotion
          if (move.to <= BLACK_PROMOTION_END ||
              move.from <= BLACK_PROMOTION_END) {
            move.promotion = 1;
            movesArray[moveNumber] = move;
            moveNumber++;
          }
        }
      }
      iterator.Init(pieces & kingRayDiagRight);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        moves = LookUpTables::getDiagRightAttacks(
                    static_cast<Square>(move.from), occupied) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          movesArray[moveNumber] = move;
          moveNumber++;
          // Promotion
          if (move.to <= BLACK_PROMOTION_END ||
              move.from <= BLACK_PROMOTION_END) {
            move.promotion = 1;
            movesArray[moveNumber] = move;
            moveNumber++;
          }
        }
      }
    }
  }

  // Rook moves
  {
    pieces = ~pinnedPieces & board[BB::Type::ROOK] &
             board[BB::Type::ALL_BLACK] & ~board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.Next()) {
      move.from = iterator.GetCurrentSquare();
      moves = (LookUpTables::getRankAttacks(static_cast<Square>(move.from),
                                            occupied) |
               LookUpTables::getFileAttacks(static_cast<Square>(move.from),
                                            occupied)) &
              validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.promotion = 0;
        movesArray[moveNumber] = move;
        moveNumber++;
        // Promotion
        if (move.to <= BLACK_PROMOTION_END ||
            move.from <= BLACK_PROMOTION_END) {
          move.promotion = 1;
          movesArray[moveNumber] = move;
          moveNumber++;
        }
      }
    }

    pieces = pinnedPieces & board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
             ~board[BB::Type::PROMOTED];
    if (pieces) {
      iterator.Init(pieces & kingRayFile);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        moves = LookUpTables::getFileAttacks(static_cast<Square>(move.from),
                                             occupied) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          movesArray[moveNumber] = move;
          moveNumber++;
          // Promotion
          if (move.to <= BLACK_PROMOTION_END ||
              move.from <= BLACK_PROMOTION_END) {
            move.promotion = 1;
            movesArray[moveNumber] = move;
            moveNumber++;
          }
        }
      }
      iterator.Init(pieces & kingRayRank);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        moves = LookUpTables::getRankAttacks(static_cast<Square>(move.from),
                                             occupied) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          movesArray[moveNumber] = move;
          moveNumber++;
          // Promotion
          if (move.to <= BLACK_PROMOTION_END ||
              move.from <= BLACK_PROMOTION_END) {
            move.promotion = 1;
            movesArray[moveNumber] = move;
            moveNumber++;
          }
        }
      }
    }
  }

  move.promotion = 0;
  // Horse moves
  {
    pieces = ~pinnedPieces & board[BB::Type::BISHOP] &
             board[BB::Type::ALL_BLACK] & board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.Next()) {
      move.from = iterator.GetCurrentSquare();
      Bitboard horse(static_cast<Square>(move.from));
      moves = (LookUpTables::getDiagRightAttacks(static_cast<Square>(move.from),
                                                 occupied) |
               LookUpTables::getDiagLeftAttacks(static_cast<Square>(move.from),
                                                occupied) |
               moveN(horse) | moveE(horse) | moveS(horse) | moveW(horse)) &
              validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        movesArray[moveNumber] = move;
        moveNumber++;
      }
    }

    pieces = pinnedPieces & board[BB::Type::BISHOP] &
             board[BB::Type::ALL_BLACK] & board[BB::Type::PROMOTED];
    if (pieces) {
      iterator.Init(pieces & kingRayDiagLeft);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        moves = LookUpTables::getDiagLeftAttacks(static_cast<Square>(move.from),
                                                 occupied) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          movesArray[moveNumber] = move;
          moveNumber++;
        }
      }
      iterator.Init(pieces & kingRayDiagRight);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        moves = LookUpTables::getDiagRightAttacks(
                    static_cast<Square>(move.from), occupied) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          movesArray[moveNumber] = move;
          moveNumber++;
        }
      }
      iterator.Init(pieces & kingRayFile);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        Bitboard horse(static_cast<Square>(move.from));
        moves = (moveN(horse) | moveS(horse)) & validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          movesArray[moveNumber] = move;
          moveNumber++;
        }
      }
      iterator.Init(pieces & kingRayRank);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        Bitboard horse(static_cast<Square>(move.from));
        moves = (moveE(horse) | moveW(horse)) & validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          movesArray[moveNumber] = move;
          moveNumber++;
        }
      }
    }
  }

  // Dragon moves
  {
    pieces = ~pinnedPieces & board[BB::Type::ROOK] &
             board[BB::Type::ALL_BLACK] & board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.Next()) {
      move.from = iterator.GetCurrentSquare();
      Bitboard dragon(static_cast<Square>(move.from));
      moves =
          (LookUpTables::getRankAttacks(static_cast<Square>(move.from),
                                        occupied) |
           LookUpTables::getFileAttacks(static_cast<Square>(move.from),
                                        occupied) |
           moveNW(dragon) | moveNE(dragon) | moveSE(dragon) | moveSW(dragon)) &
          validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        movesArray[moveNumber] = move;
        moveNumber++;
      }
    }

    pieces = pinnedPieces & board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
             board[BB::Type::PROMOTED];
    if (pieces) {
      iterator.Init(pieces & kingRayFile);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        moves = LookUpTables::getFileAttacks(static_cast<Square>(move.from),
                                             occupied) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          movesArray[moveNumber] = move;
          moveNumber++;
        }
      }
      iterator.Init(pieces & kingRayRank);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        moves = LookUpTables::getRankAttacks(static_cast<Square>(move.from),
                                             occupied) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          movesArray[moveNumber] = move;
          moveNumber++;
        }
      }
      iterator.Init(pieces & kingRayDiagRight);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        Bitboard dragon(static_cast<Square>(move.from));
        moves = (moveNE(dragon) | moveSW(dragon)) & validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          movesArray[moveNumber] = move;
          moveNumber++;
        }
      }
      iterator.Init(pieces & kingRayDiagLeft);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        Bitboard dragon(static_cast<Square>(move.from));
        moves = (moveNW(dragon) | moveSE(dragon)) & validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          movesArray[moveNumber] = move;
          moveNumber++;
        }
      }
    }
  }

  // Drop moves
  {
    Bitboard legalDropSpots;
    // Pawns
    if (board.inHand.pieceNumber.BlackPawn > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last rank
      legalDropSpots[TOP] &= ~TOP_RANK;
      // Cannot drop to give checkmate
      pieces = board[BB::Type::KING] & board[BB::Type::ALL_WHITE];
      // All valid enemy king moves
      moves =
          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
          ~ourAttacks & ~board[BB::Type::ALL_WHITE];
      // If there is only one spot pawn cannot block it
      if (popcount(moves[TOP]) + popcount(moves[MID]) +
              popcount(moves[BOTTOM]) ==
          1) {
        legalDropSpots &= ~moveS(moves);
      }
      // Cannot drop on file with other pawn
      Bitboard validFiles;
      for (int fileIdx = 0; fileIdx < BOARD_DIM; fileIdx++) {
        Bitboard file = getFullFile(fileIdx);
        if (!(file & board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
              ~board[BB::Type::PROMOTED])) {
          validFiles |= file;
        }
      }
      legalDropSpots &= validFiles;
      move.from = BLACK_PAWN_DROP;
      movesIterator.Init(legalDropSpots);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        movesArray[moveNumber] = move;
        moveNumber++;
      }
    }
    if (board.inHand.pieceNumber.BlackLance > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last rank
      legalDropSpots[TOP] &= ~TOP_RANK;
      move.from = BLACK_LANCE_DROP;
      movesIterator.Init(legalDropSpots);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        movesArray[moveNumber] = move;
        moveNumber++;
      }
    }
    if (board.inHand.pieceNumber.BlackKnight > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last two ranks
      legalDropSpots[TOP] &= BOTTOM_RANK;
      move.from = BLACK_KNIGHT_DROP;
      movesIterator.Init(legalDropSpots);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        movesArray[moveNumber] = move;
        moveNumber++;
      }
    }
    legalDropSpots = validMoves & ~occupied;
    movesIterator.Init(legalDropSpots);
    while (movesIterator.Next()) {
      if (board.inHand.pieceNumber.BlackSilverGeneral > 0) {
        move.from = BLACK_SILVER_GENERAL_DROP;
        move.to = movesIterator.GetCurrentSquare();
        movesArray[moveNumber] = move;
        moveNumber++;
      }
      if (board.inHand.pieceNumber.BlackGoldGeneral > 0) {
        move.from = BLACK_GOLD_GENERAL_DROP;
        move.to = movesIterator.GetCurrentSquare();
        movesArray[moveNumber] = move;
        moveNumber++;
      }
      if (board.inHand.pieceNumber.BlackBishop > 0) {
        move.from = BLACK_BISHOP_DROP;
        move.to = movesIterator.GetCurrentSquare();
        movesArray[moveNumber] = move;
        moveNumber++;
      }
      if (board.inHand.pieceNumber.BlackRook > 0) {
        move.from = BLACK_ROOK_DROP;
        move.to = movesIterator.GetCurrentSquare();
        movesArray[moveNumber] = move;
        moveNumber++;
      }
    }
  }

  return moveNumber;
}
}  // namespace engine
}  // namespace shogi