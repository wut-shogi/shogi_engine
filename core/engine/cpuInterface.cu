#include "MoveGenHelpers.h"
#include "cpuInterface.h"

namespace shogi {
namespace engine {
void CPU::countWhiteMoves(Board* inBoards,
                          uint32_t inBoardsLength,
                          Bitboard* outValidMoves,
                          Bitboard* outAttackedByEnemy,
                          Bitboard* outPinned,
                          uint32_t* outMovesOffset,
                          bool* isMate) {
  *isMate = false;
  for (int index = 0; index < inBoardsLength; index++) {
    Board board = inBoards[index];
    Bitboard king = board[BB::Type::KING] & board[BB::Type::ALL_WHITE];
    Bitboard notPromoted = ~board[BB::Type::PROMOTED];
    Bitboard checkingPieces, attacked, pieces, moves, slidingChecksPaths,
        attacks, attacksFull, mask, potentialPin, pinned, ourAttacks;
    BitboardIterator iterator;
    Square square;
    size_t numberOfMoves = 0;

    // Non Sliding pieces
    // Pawns
    pieces = board.bbs[BB::Type::PAWN] & board.bbs[BB::Type::ALL_BLACK] &
             notPromoted;
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
    checkingPieces |= (moveSE(king) | moveS(king) | moveSW(king) |
                       moveNE(king) | moveNW(king)) &
                      pieces;
    attacked |= moveNE(pieces) | moveN(pieces) | moveNW(pieces) |
                moveSE(pieces) | moveSW(pieces);
    // Gold generals
    pieces = (board[BB::Type::GOLD_GENERAL] |
              ((board[BB::Type::PAWN] | board[BB::Type::KNIGHT] |
                board[BB::Type::SILVER_GENERAL] | board[BB::Type::LANCE]) &
               board[BB::Type::PROMOTED])) &
             board.bbs[BB::Type::ALL_BLACK];
    checkingPieces |= (moveSE(king) | moveS(king) | moveSW(king) | moveE(king) |
                       moveW(king) | moveN(king)) &
                      pieces;
    attacked |= moveNE(pieces) | moveN(pieces) | moveNW(pieces) |
                moveE(pieces) | moveW(pieces) | moveS(pieces);
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
    attacked |=
        moveNW(pieces) | moveNE(pieces) | moveSE(pieces) | moveSW(pieces);

    // Sliding pieces
    iterator.Init(king);
    iterator.Next();
    Square kingSquare = iterator.GetCurrentSquare();
    Bitboard occupied = board[BB::Type::ALL_BLACK] | board[BB::Type::ALL_WHITE];
    // Lance
    {
      pieces =
          board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] & notPromoted;
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
        mask = getFileMask(squareToFile(square)) &
               getRankMask(squareToRank(square));
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

    int numberOfCheckingPieces = popcount(checkingPieces[TOP]) +
                                 popcount(checkingPieces[MID]) +
                                 popcount(checkingPieces[BOTTOM]);

    // King can always move to non attacked squares
    moves = moveNE(king) | moveN(king) | moveNW(king) | moveE(king) |
            moveW(king) | moveSE(king) | moveS(king) | moveSW(king);
    moves &= ~attacked & ~board[BB::Type::ALL_WHITE];
    numberOfMoves +=
        popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);

    Bitboard validMoves;
    // If more then one piece is checking the king and king cannot move its mate
    if (numberOfCheckingPieces > 1) {
      if (numberOfMoves == 0) {
        *isMate = true;
        return;
      }
    } else if (numberOfCheckingPieces == 1) {
      // if king is checked by exactly one piece legal moves can also be block
      // sliding check or capture a checking piece
      validMoves = checkingPieces | (slidingChecksPaths & ~king);
    } else if (numberOfCheckingPieces == 0) {
      // If there is no checks all moves are valid (you cannot capture your own
      // piece)
      validMoves = ~board[BB::Type::ALL_WHITE];
    }

    outValidMoves[index] = validMoves;
    outAttackedByEnemy[index] = attacked;
    outPinned[index] = pinned;

    // Pawn moves
    {
      pieces = ~pinned & board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
               notPromoted;
      moves = moveS(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) +
          popcount(moves[BOTTOM] & ~BOTTOM_RANK) * 2 +  // promotions
          popcount(moves[BOTTOM] & BOTTOM_RANK);        // forced promotion
    }

    // Knight moves
    {
      pieces = ~pinned & board[BB::Type::KNIGHT] & board[BB::Type::ALL_WHITE] &
               notPromoted;
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
      pieces = ~pinned & board[BB::Type::SILVER_GENERAL] &
               board[BB::Type::ALL_WHITE] & notPromoted;
      moves = moveS(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += popcount(moves[TOP]) + popcount(moves[MID]) +
                       popcount(moves[BOTTOM]) * 2;  // promotions
      moves = moveSE(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += popcount(moves[TOP]) + popcount(moves[MID]) +
                       popcount(moves[BOTTOM]) * 2;  // promotions
      moves = moveSW(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += popcount(moves[TOP]) + popcount(moves[MID]) +
                       popcount(moves[BOTTOM]) * 2;  // promotions
      moves = moveNE(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += popcount(moves[TOP]) +
                       popcount(moves[MID] & BOTTOM_RANK) *
                           2 +  // promotion when starting from promotion zone
                       popcount(moves[MID] & ~BOTTOM_RANK) +
                       popcount(moves[BOTTOM]) * 2;  // promotions
      moves = moveNW(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += popcount(moves[TOP]) +
                       popcount(moves[MID] & BOTTOM_RANK) *
                           2 +  // promotion when starting from promotion zone
                       popcount(moves[MID] & ~BOTTOM_RANK) +
                       popcount(moves[BOTTOM]) * 2;  // promotions
    }

    // GoldGenerals moves
    {
      pieces = ~pinned &
               (board[BB::Type::GOLD_GENERAL] |
                ((board[BB::Type::PAWN] | board[BB::Type::LANCE] |
                  board[BB::Type::KNIGHT] | board[BB::Type::SILVER_GENERAL]) &
                 board[BB::Type::PROMOTED])) &
               board[BB::Type::ALL_WHITE];
      moves = moveS(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveSE(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveSW(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveE(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveW(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveN(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
    }

    // Lance moves
    {
      pieces = ~pinned & board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] &
               notPromoted;
      iterator.Init(pieces);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        moves = getFileAttacks(square, occupied) &
                ~getRankMask(squareToRank(square)) & validMoves;
        ourAttacks |= moves;
        numberOfMoves +=
            popcount(moves[TOP]) + popcount(moves[MID]) +
            popcount(moves[BOTTOM] & ~BOTTOM_RANK) * 2 +  // promotions
            popcount(moves[BOTTOM] & BOTTOM_RANK);        // forced promotion
      }
    }

    // Bishop moves
    {
      pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
               notPromoted;
      iterator.Init(pieces);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        moves = (getDiagRightAttacks(square, occupied) |
                 getDiagLeftAttacks(square, occupied)) &
                validMoves;
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

    // Rook moves
    {
      pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
               notPromoted;
      iterator.Init(pieces);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        moves = (getRankAttacks(square, occupied) |
                 getFileAttacks(square, occupied)) &
                validMoves;
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

    // Horse moves
    {
      pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
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
        numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                          popcount(moves[BOTTOM]));
      }
    }

    // Dragon moves
    {
      pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
               board[BB::Type::PROMOTED];
      iterator.Init(pieces);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        Bitboard dragon(square);
        moves = (getRankAttacks(square, occupied) |
                 getFileAttacks(square, occupied) | moveNW(dragon) |
                 moveNE(dragon) | moveSE(dragon) | moveSW(dragon)) &
                validMoves;
        ourAttacks |= moves;
        numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                          popcount(moves[BOTTOM]));
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
                notPromoted)) {
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
    if (numberOfMoves == 0) {
      *isMate = true;
      return;
    }
    outMovesOffset[index] = numberOfMoves;
  }
}

void CPU::countBlackMoves(Board* inBoards,
                          uint32_t inBoardsLength,
                          Bitboard* outValidMoves,
                          Bitboard* outAttackedByEnemy,
                          Bitboard* outPinned,
                          uint32_t* outMovesOffset,
                          bool* isMate) {
  *isMate == false;
  for (int index = 0; index < inBoardsLength; index++) {
    Board board = inBoards[index];
    Bitboard king = board[BB::Type::KING] & board[BB::Type::ALL_BLACK];
    Bitboard notPromoted = ~board[BB::Type::PROMOTED];
    Bitboard checkingPieces, attacked, pieces, moves, slidingChecksPaths,
        attacks, attacksFull, mask, potentialPin, pinned, ourAttacks;
    BitboardIterator iterator;
    Square square;
    size_t numberOfMoves = 0;

    // Non Sliding pieces
    // Pawns
    pieces = board.bbs[BB::Type::PAWN] & board.bbs[BB::Type::ALL_WHITE] &
             notPromoted;
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
    checkingPieces |= (moveNE(king) | moveN(king) | moveNW(king) |
                       moveSE(king) | moveSW(king)) &
                      pieces;
    attacked |= moveSE(pieces) | moveS(pieces) | moveSW(pieces) |
                moveNE(pieces) | moveNW(pieces);
    // gold generals
    pieces = (board[BB::Type::GOLD_GENERAL] |
              ((board[BB::Type::PAWN] | board[BB::Type::KNIGHT] |
                board[BB::Type::SILVER_GENERAL] | board[BB::Type::LANCE]) &
               board[BB::Type::PROMOTED])) &
             board.bbs[BB::Type::ALL_WHITE];
    checkingPieces |= (moveNE(king) | moveN(king) | moveNW(king) | moveE(king) |
                       moveW(king) | moveS(king)) &
                      pieces;
    attacked |= moveSE(pieces) | moveS(pieces) | moveSW(pieces) |
                moveE(pieces) | moveW(pieces) | moveN(pieces);
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
    attacked |=
        moveNW(pieces) | moveNE(pieces) | moveSE(pieces) | moveSW(pieces);

    // Sliding pieces
    iterator.Init(king);
    iterator.Next();
    Square kingSquare = iterator.GetCurrentSquare();
    Bitboard occupied = board[BB::Type::ALL_BLACK] | board[BB::Type::ALL_WHITE];
    // Lance
    {
      pieces =
          board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] & notPromoted;
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
        mask = getFileMask(squareToFile(square)) &
               getRankMask(squareToRank(square));
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

    int numberOfCheckingPieces = popcount(checkingPieces[TOP]) +
                                 popcount(checkingPieces[MID]) +
                                 popcount(checkingPieces[BOTTOM]);

    // King can always move to non attacked squares
    moves = moveNE(king) | moveN(king) | moveNW(king) | moveE(king) |
            moveW(king) | moveSE(king) | moveS(king) | moveSW(king);
    moves &= ~attacked & ~board[BB::Type::ALL_BLACK];
    numberOfMoves +=
        popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
    Bitboard validMoves;
    // If more then one piece is checking the king and king cannot move its mate
    if (numberOfCheckingPieces > 1) {
      if (numberOfMoves == 0) {
        *isMate = true;
        return;
      }
    } else if (numberOfCheckingPieces == 1) {
      // if king is checked by exactly one piece legal moves can also be block
      // sliding check or capture a checking piece
      validMoves = checkingPieces | (slidingChecksPaths & ~king);
    } else if (numberOfCheckingPieces == 0) {
      // If there is no checks all moves are valid (you cannot capture your own
      // piece)
      validMoves = ~board[BB::Type::ALL_BLACK];
    }

    outValidMoves[index] = validMoves;
    outAttackedByEnemy[index] = attacked;
    outPinned[index] = pinned;

    // Pawn moves
    {
      pieces = ~pinned & board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
               notPromoted;
      moves = moveN(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += popcount(moves[TOP] & TOP_RANK) +  // forced promotions
                       popcount(moves[TOP] & ~TOP_RANK) * 2 +  // promotions
                       popcount(moves[MID]) + popcount(moves[BOTTOM]);
    }

    // Knight moves
    {
      pieces = ~pinned & board[BB::Type::KNIGHT] & board[BB::Type::ALL_BLACK] &
               notPromoted;
      moves = moveN(moveNE(pieces)) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP] & ~BOTTOM_RANK) +     // forced promotions
          popcount(moves[TOP] & BOTTOM_RANK) * 2 +  // promotions
          popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveN(moveNW(pieces)) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += popcount(moves[TOP] & ~TOP_RANK) +  // forced promotions
                       popcount(moves[TOP] & TOP_RANK) * 2 +  // promotions
                       popcount(moves[MID]) + popcount(moves[BOTTOM]);
    }

    // SilverGenerals moves
    {
      pieces = ~pinned & board[BB::Type::SILVER_GENERAL] &
               board[BB::Type::ALL_BLACK] & notPromoted;
      moves = moveN(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += popcount(moves[TOP]) * 2 +  // promotions
                       popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveNE(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += popcount(moves[TOP]) * 2 +  // promotions
                       popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveNW(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += popcount(moves[TOP]) * 2 +  // promotions
                       popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveSE(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += popcount(moves[TOP]) * 2 +  // promotions
                       popcount(moves[MID] & TOP_RANK) *
                           2 +  // promotion when starting from promotion zone
                       popcount(moves[MID] & ~TOP_RANK) +
                       popcount(moves[BOTTOM]);
      moves = moveSW(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += popcount(moves[TOP]) * 2 +  // promotions
                       popcount(moves[MID] & TOP_RANK) *
                           2 +  // promotion when starting from promotion zone
                       popcount(moves[MID] & ~TOP_RANK) +
                       popcount(moves[BOTTOM]);
    }

    // GoldGenerals moves
    {
      pieces = ~pinned &
               (board[BB::Type::GOLD_GENERAL] |
                ((board[BB::Type::PAWN] | board[BB::Type::LANCE] |
                  board[BB::Type::KNIGHT] | board[BB::Type::SILVER_GENERAL]) &
                 board[BB::Type::PROMOTED])) &
               board[BB::Type::ALL_BLACK];
      moves = moveN(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveNE(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveNW(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveE(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveW(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveS(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
    }

    // Lance moves
    {
      pieces = ~pinned & board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] &
               notPromoted;
      iterator.Init(pieces);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        moves = getFileAttacks(square, occupied) &
                getRankMask(squareToRank(square)) & validMoves;
        ourAttacks |= moves;
        numberOfMoves += popcount(moves[TOP] & TOP_RANK) +  // forced promotions
                         popcount(moves[TOP] & ~TOP_RANK) * 2 +  // promotions
                         popcount(moves[MID]) + popcount(moves[BOTTOM]);
      }
    }

    // Bishop moves
    {
      pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
               notPromoted;
      iterator.Init(pieces);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        moves = (getDiagRightAttacks(square, occupied) |
                 getDiagLeftAttacks(square, occupied)) &
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
    }

    // Rook moves
    {
      pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
               notPromoted;
      iterator.Init(pieces);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        moves = (getRankAttacks(square, occupied) |
                 getFileAttacks(square, occupied)) &
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
    }

    // Horse moves
    {
      pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
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
        numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                          popcount(moves[BOTTOM]));
      }
    }

    // Dragon moves
    {
      pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
               board[BB::Type::PROMOTED];
      iterator.Init(pieces);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        Bitboard dragon(square);
        moves = (getRankAttacks(square, occupied) |
                 getFileAttacks(square, occupied) | moveNW(dragon) |
                 moveNE(dragon) | moveSE(dragon) | moveSW(dragon)) &
                validMoves;
        ourAttacks |= moves;
        numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                          popcount(moves[BOTTOM]));
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
                notPromoted)) {
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
    if (numberOfMoves == 0) {
      *isMate = true;
      return;
    }
    outMovesOffset[index] = numberOfMoves;
  }
}

void CPU::generateWhiteMoves(Board* inBoards,
                             uint32_t inBoardsLength,
                             Bitboard* inValidMoves,
                             Bitboard* inAttackedByEnemy,
                             Bitboard* inPinned,
                             uint32_t* inMovesOffset,
                             Move* outMoves,
                             uint32_t* outMoveToBoardIdx) {
  for (int index = 0; index < inBoardsLength; index++) {
    Board board = inBoards[index];
    Bitboard validMoves = inValidMoves[index];
    Bitboard notPromoted = ~board[BB::Type::PROMOTED];
    Bitboard pinned = inPinned[index];
    Bitboard pieces, moves, ourAttacks;
    Bitboard occupied = board[BB::Type::ALL_WHITE] | board[BB::Type::ALL_BLACK];
    BitboardIterator movesIterator, iterator;
    Move move;
    uint32_t movesCount = inMovesOffset[index + 1] - inMovesOffset[index];
    uint32_t movesOffset = inMovesOffset[index];
    uint32_t moveNumber = 0;
    // Pawn moves
    {
      pieces = ~pinned & board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
               notPromoted;
      moves = moveS(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + N;
        // Promotion
        if (move.to >= WHITE_PROMOTION_START) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
        // Not when forced promotion
        if (move.to < WHITE_PAWN_LANCE_FORECED_PROMOTION_START) {
          move.promotion = 0;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
      }
    }
    // Knight moves
    {
      pieces = ~pinned & board[BB::Type::KNIGHT] & board[BB::Type::ALL_WHITE] &
               notPromoted;
      moves = moveS(moveSE(pieces)) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + N + NW;
        // Promotion
        if (move.to >= WHITE_PROMOTION_START) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
        // Not when forced promotion
        if (move.to < WHITE_HORSE_FORCED_PROMOTION_START) {
          move.promotion = 0;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
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
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
        // Not when forced promotion
        if (move.to < WHITE_HORSE_FORCED_PROMOTION_START) {
          move.promotion = 0;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
      }
    }

    // SilverGenerals moves
    {
      pieces = ~pinned & board[BB::Type::SILVER_GENERAL] &
               board[BB::Type::ALL_WHITE] & notPromoted;
      moves = moveS(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + N;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
        // Promotion
        if (move.to >= WHITE_PROMOTION_START) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
      }
      moves = moveSE(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + NW;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
        // Promotion
        if (move.to >= WHITE_PROMOTION_START) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
      }
      moves = moveSW(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + NE;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
        // Promotion
        if (move.to >= WHITE_PROMOTION_START) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
      }
      moves = moveNE(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + SW;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
        // Promotion
        if (move.to >= WHITE_PROMOTION_START ||
            move.from >= WHITE_PROMOTION_START) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
      }
      moves = moveNW(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + SE;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
        // Promotion
        if (move.to >= WHITE_PROMOTION_START ||
            move.from >= WHITE_PROMOTION_START) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
      }
    }

    // GoldGenerals moves
    {
      pieces = ~pinned &
               (board[BB::Type::GOLD_GENERAL] |
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
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
      moves = moveSE(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + NW;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
      moves = moveSW(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + NE;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
      moves = moveE(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + W;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
      moves = moveW(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + E;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
      moves = moveN(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + S;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }

    // Lances moves
    {
      pieces = ~pinned & board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] &
               notPromoted;
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
            outMoves[movesOffset + moveNumber] = move;

            outMoveToBoardIdx[movesOffset + moveNumber] = index;
            moveNumber++;
          }
          // Not when forced promotion
          if (move.to < WHITE_PAWN_LANCE_FORECED_PROMOTION_START) {
            move.promotion = 0;
            outMoves[movesOffset + moveNumber] = move;

            outMoveToBoardIdx[movesOffset + moveNumber] = index;
            moveNumber++;
          }
        }
      }
    }

    // Bishop moves
    {
      pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
               notPromoted;
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
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
          // Promotion
          if (move.to >= WHITE_PROMOTION_START ||
              move.from >= WHITE_PROMOTION_START) {
            move.promotion = 1;
            outMoves[movesOffset + moveNumber] = move;

            outMoveToBoardIdx[movesOffset + moveNumber] = index;
            moveNumber++;
          }
        }
      }
    }

    // Rook moves
    {
      pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
               notPromoted;
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
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
          // Promotion
          if (move.to >= WHITE_PROMOTION_START ||
              move.from >= WHITE_PROMOTION_START) {
            move.promotion = 1;
            outMoves[movesOffset + moveNumber] = move;

            outMoveToBoardIdx[movesOffset + moveNumber] = index;
            moveNumber++;
          }
        }
      }
    }
    move.promotion = 0;

    // Horse moves
    {
      pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
               board[BB::Type::PROMOTED];
      iterator.Init(pieces);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        Bitboard horse(static_cast<Square>(move.from));
        moves = (getDiagRightAttacks(static_cast<Square>(move.from), occupied) |
                 getDiagLeftAttacks(static_cast<Square>(move.from), occupied) |
                 moveN(horse) | moveE(horse) | moveS(horse) | moveW(horse)) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
      }
    }

    // Dragon moves
    {
      pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
               board[BB::Type::PROMOTED];
      iterator.Init(pieces);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        Bitboard dragon(static_cast<Square>(move.from));
        moves = (getRankAttacks(static_cast<Square>(move.from), occupied) |
                 getFileAttacks(static_cast<Square>(move.from), occupied) |
                 moveNW(pieces) | moveNE(pieces) | moveSE(pieces) |
                 moveSW(pieces)) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
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
            ~inAttackedByEnemy[index] & ~board[BB::Type::ALL_WHITE];
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
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
                notPromoted)) {
            validFiles |= file;
          }
        }
        legalDropSpots &= validFiles;
        movesIterator.Init(legalDropSpots);
        move.from = WHITE_PAWN_DROP;
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
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
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
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
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
      }
      legalDropSpots = validMoves & ~occupied;
      movesIterator.Init(legalDropSpots);
      while (movesIterator.Next()) {
        if (board.inHand.pieceNumber.WhiteSilverGeneral > 0) {
          move.from = WHITE_SILVER_GENERAL_DROP;
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
        if (board.inHand.pieceNumber.WhiteGoldGeneral > 0) {
          move.from = WHITE_GOLD_GENERAL_DROP;
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
        if (board.inHand.pieceNumber.WhiteBishop > 0) {
          move.from = WHITE_BISHOP_DROP;
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
        if (board.inHand.pieceNumber.WhiteRook > 0) {
          move.from = WHITE_ROOK_DROP;
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
      }
    }

    if (moveNumber > movesCount) {
      std::cout << "Error in:" << std::endl;
      std::cout << boardToSFEN(board) << std::endl;
    }
  }
}

void CPU::generateBlackMoves(Board* inBoards,
                             uint32_t inBoardsLength,
                             Bitboard* inValidMoves,
                             Bitboard* inAttackedByEnemy,
                             Bitboard* inPinned,
                             uint32_t* inMovesOffset,
                             Move* outMoves,
                             uint32_t* outMoveToBoardIdx) {
  for (int index = 0; index < inBoardsLength; index++) {
    Board board = inBoards[index];
    Bitboard validMoves = inValidMoves[index];
    Bitboard notPromoted = ~board[BB::Type::PROMOTED];
    Bitboard pinned = inPinned[index];
    Bitboard pieces, moves, ourAttacks;
    Bitboard occupied = board[BB::Type::ALL_WHITE] | board[BB::Type::ALL_BLACK];
    BitboardIterator movesIterator, iterator;
    Move move;
    uint32_t movesCount = inMovesOffset[index + 1] - inMovesOffset[index];
    uint32_t movesOffset = inMovesOffset[index];
    uint32_t moveNumber = 0;
    // Pawn moves
    {
      pieces = ~pinned & board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
               notPromoted;
      moves = moveN(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + S;
        // Promotion
        if (move.to <= BLACK_PROMOTION_END) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
        // Not when forced promotion
        if (move.to > BLACK_PAWN_LANCE_FORCE_PROMOTION_END) {
          move.promotion = 0;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
      }
    }
    // Knight moves
    {
      pieces = ~pinned & board[BB::Type::KNIGHT] & board[BB::Type::ALL_BLACK] &
               notPromoted;
      moves = moveN(moveNE(pieces)) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + S + SW;
        // Promotion
        if (move.to <= BLACK_PROMOTION_END) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
        // Not when forced promotion
        if (move.to > BLACK_HORSE_FORCED_PROMOTION_END) {
          move.promotion = 0;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
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
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
        // Not when forced promotion
        if (move.to > BLACK_HORSE_FORCED_PROMOTION_END) {
          move.promotion = 0;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
      }
    }
    // SilverGenerals moves
    {
      pieces = ~pinned & board[BB::Type::SILVER_GENERAL] &
               board[BB::Type::ALL_BLACK] & notPromoted;
      moves = moveN(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + S;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
        // Promotion
        if (move.to <= BLACK_PROMOTION_END) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
      }
      moves = moveNE(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + SW;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
        // Promotion
        if (move.to <= BLACK_PROMOTION_END) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
      }
      moves = moveNW(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + SE;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
        // Promotion
        if (move.to <= BLACK_PROMOTION_END) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
      }
      moves = moveSE(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + NW;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
        // Promotion
        if (move.to <= BLACK_PROMOTION_END ||
            move.from <= BLACK_PROMOTION_END) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
      }
      moves = moveSW(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + NE;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
        // Promotion
        if (move.to <= BLACK_PROMOTION_END ||
            move.from <= BLACK_PROMOTION_END) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
      }
    }
    // GoldGenerals moves
    {
      pieces = ~pinned &
               (board[BB::Type::GOLD_GENERAL] |
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
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
      moves = moveNE(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + SW;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
      moves = moveNW(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + SE;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
      moves = moveE(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + W;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
      moves = moveW(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + E;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
      moves = moveS(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + N;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
    // Lances moves
    {
      pieces = ~pinned & board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] &
               notPromoted;
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
            outMoves[movesOffset + moveNumber] = move;

            outMoveToBoardIdx[movesOffset + moveNumber] = index;
            moveNumber++;
          }
          // Not when forced promotion
          if (move.to > BLACK_PAWN_LANCE_FORCE_PROMOTION_END) {
            move.promotion = 0;
            outMoves[movesOffset + moveNumber] = move;

            outMoveToBoardIdx[movesOffset + moveNumber] = index;
            moveNumber++;
          }
        }
      }
    }
    // Bishop moves
    {
      pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
               notPromoted;
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
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
          // Promotion
          if (move.to <= BLACK_PROMOTION_END ||
              move.from <= BLACK_PROMOTION_END) {
            move.promotion = 1;
            outMoves[movesOffset + moveNumber] = move;

            outMoveToBoardIdx[movesOffset + moveNumber] = index;
            moveNumber++;
          }
        }
      }
    }
    // Rook moves
    {
      pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
               notPromoted;
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
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
          // Promotion
          if (move.to <= BLACK_PROMOTION_END ||
              move.from <= BLACK_PROMOTION_END) {
            move.promotion = 1;
            outMoves[movesOffset + moveNumber] = move;

            outMoveToBoardIdx[movesOffset + moveNumber] = index;
            moveNumber++;
          }
        }
      }
    }
    move.promotion = 0;
    // Horse moves
    {
      pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
               board[BB::Type::PROMOTED];
      iterator.Init(pieces);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        Bitboard horse(static_cast<Square>(move.from));
        moves = (getDiagRightAttacks(static_cast<Square>(move.from), occupied) |
                 getDiagLeftAttacks(static_cast<Square>(move.from), occupied) |
                 moveN(horse) | moveE(horse) | moveS(horse) | moveW(horse)) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
      }
    }
    // Dragon moves
    {
      pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
               board[BB::Type::PROMOTED];
      iterator.Init(pieces);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        Bitboard dragon(static_cast<Square>(move.from));
        moves = (getRankAttacks(static_cast<Square>(move.from), occupied) |
                 getFileAttacks(static_cast<Square>(move.from), occupied) |
                 moveNW(dragon) | moveNE(dragon) | moveSE(dragon) |
                 moveSW(dragon)) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
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
            ~inAttackedByEnemy[index] & ~board[BB::Type::ALL_BLACK];
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
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
                notPromoted)) {
            validFiles |= file;
          }
        }
        legalDropSpots &= validFiles;
        move.from = BLACK_PAWN_DROP;
        movesIterator.Init(legalDropSpots);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
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
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
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
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
      }
      legalDropSpots = validMoves & ~occupied;
      movesIterator.Init(legalDropSpots);
      while (movesIterator.Next()) {
        if (board.inHand.pieceNumber.BlackSilverGeneral > 0) {
          move.from = BLACK_SILVER_GENERAL_DROP;
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
        if (board.inHand.pieceNumber.BlackGoldGeneral > 0) {
          move.from = BLACK_GOLD_GENERAL_DROP;
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
        if (board.inHand.pieceNumber.BlackBishop > 0) {
          move.from = BLACK_BISHOP_DROP;
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
        if (board.inHand.pieceNumber.BlackRook > 0) {
          move.from = BLACK_ROOK_DROP;
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
      }
    }

    if (moveNumber > movesCount) {
      std::cout << "error" << std::endl;
    }
  }
}

void CPU::generateWhiteBoards(Move* inMoves,
                              uint32_t inMovesLength,
                              Board* inBoards,
                              uint32_t* moveToBoardIdx,
                              Board* outBoards) {
  uint64_t one = 1;
  for (int index = 0; index < inMovesLength; index++) {
    Board board = inBoards[moveToBoardIdx[index]];
    Move move = inMoves[index];
    Region toRegionIdx = squareToRegion(static_cast<Square>(move.to));
    uint32_t toRegion = 1 << (REGION_SIZE - 1 - move.to % REGION_SIZE);
    if (move.from < SQUARE_SIZE) {
      Region fromRegionIdx = squareToRegion(static_cast<Square>(move.from));
      uint32_t fromRegion = 1 << (REGION_SIZE - 1 - move.from % REGION_SIZE);
      for (int i = 0; i < BB::Type::KING; i++) {
        if (board[static_cast<BB::Type>(i)][toRegionIdx] & toRegion) {
          board[static_cast<BB::Type>(i)][toRegionIdx] &= ~toRegion;
          uint64_t addedValue = one << (4 * i);
          board.inHand.value += addedValue;
        }
        if (board[static_cast<BB::Type>(i)][fromRegionIdx] & fromRegion) {
          board[static_cast<BB::Type>(i)][fromRegionIdx] &= ~fromRegion;
          board[static_cast<BB::Type>(i)][toRegionIdx] |= toRegion;
        }
      }
      for (int i = BB::Type::KING; i < BB::Type::SIZE; i++) {
        if (board[static_cast<BB::Type>(i)][toRegionIdx] & toRegion) {
          board[static_cast<BB::Type>(i)][toRegionIdx] &= ~toRegion;
        }
        if (board[static_cast<BB::Type>(i)][fromRegionIdx] & fromRegion) {
          board[static_cast<BB::Type>(i)][fromRegionIdx] &= ~fromRegion;
          board[static_cast<BB::Type>(i)][toRegionIdx] |= toRegion;
        }
      }
      if (move.promotion) {
        board[BB::Type::PROMOTED][toRegionIdx] |= toRegion;
      }
    } else {
      int offset = move.from - WHITE_PAWN_DROP;
      uint64_t addedValue = one << (4 * offset);
      board.inHand.value -= addedValue;
      board[static_cast<BB::Type>(offset % 7)][toRegionIdx] |= toRegion;
      board[static_cast<BB::Type>(BB::Type::ALL_WHITE)][toRegionIdx] |=
          toRegion;
    }
    outBoards[index] = board;
  }
}

void CPU::generateBlackBoards(Move* inMoves,
                              uint32_t inMovesLength,
                              Board* inBoards,
                              uint32_t* moveToBoardIdx,
                              Board* outBoards) {
  uint64_t one = 1;
  for (int index = 0; index < inMovesLength; index++) {
    Board board = inBoards[moveToBoardIdx[index]];
    Move move = inMoves[index];
    Region toRegionIdx = squareToRegion(static_cast<Square>(move.to));
    uint32_t toRegion = 1 << (REGION_SIZE - 1 - move.to % REGION_SIZE);
    if (move.from < SQUARE_SIZE) {
      Region fromRegionIdx = squareToRegion(static_cast<Square>(move.from));
      uint32_t fromRegion = 1 << (REGION_SIZE - 1 - move.from % REGION_SIZE);
      for (int i = 0; i < BB::Type::KING; i++) {
        if (board[static_cast<BB::Type>(i)][toRegionIdx] & toRegion) {
          board[static_cast<BB::Type>(i)][toRegionIdx] &= ~toRegion;
          uint64_t addedValue = one << (4 * (7 + i));
          board.inHand.value += addedValue;
        }
        if (board[static_cast<BB::Type>(i)][fromRegionIdx] & fromRegion) {
          board[static_cast<BB::Type>(i)][fromRegionIdx] &= ~fromRegion;
          board[static_cast<BB::Type>(i)][toRegionIdx] |= toRegion;
        }
      }
      for (int i = BB::Type::KING; i < BB::Type::SIZE; i++) {
        if (board[static_cast<BB::Type>(i)][toRegionIdx] & toRegion) {
          board[static_cast<BB::Type>(i)][toRegionIdx] &= ~toRegion;
        }
        if (board[static_cast<BB::Type>(i)][fromRegionIdx] & fromRegion) {
          board[static_cast<BB::Type>(i)][fromRegionIdx] &= ~fromRegion;
          board[static_cast<BB::Type>(i)][toRegionIdx] |= toRegion;
        }
      }
      if (move.promotion) {
        board[BB::Type::PROMOTED][toRegionIdx] |= toRegion;
      }
    } else {
      int offset = move.from - WHITE_PAWN_DROP;
      uint64_t addedValue = one << (4 * offset);
      board.inHand.value -= addedValue;
      board[static_cast<BB::Type>(offset % 7)][toRegionIdx] |= toRegion;
      board[static_cast<BB::Type>(BB::Type::ALL_BLACK)][toRegionIdx] |=
          toRegion;
    }
    outBoards[index] = board;
  }
}

void CPU::prefixSum(uint32_t* inValues, uint32_t inValuesLength) {
  inValues[0] = 0;
  for (int index = 1; index < inValuesLength; index++) {
    inValues[index] += inValues[index - 1];
  }
}

void CPU::evaluateBoards(Board* inBoards,
                         uint32_t inBoardsLength,
                         int16_t* outValues) {
  for (int index = 1; index < inBoardsLength; index++) {
    Board board = inBoards[index];
    int16_t whitePoints = 0, blackPoints = 0;
    Bitboard pieces;
    // White
    // Pawns
    pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
             ~board[BB::Type::PROMOTED];
    whitePoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                    popcount(pieces[BOTTOM])) *
                   PieceValue::PAWN;
    pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
             board[BB::Type::PROMOTED];
    whitePoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                    popcount(pieces[BOTTOM])) *
                   PieceValue::PROMOTED_PAWN;
    whitePoints +=
        board.inHand.pieceNumber.WhitePawn * PieceValue::IN_HAND_LANCE;
    // Lances
    pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] &
             ~board[BB::Type::PROMOTED];
    whitePoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                    popcount(pieces[BOTTOM])) *
                   PieceValue::LANCE;
    pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] &
             board[BB::Type::PROMOTED];
    whitePoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                    popcount(pieces[BOTTOM])) *
                   PieceValue::PROMOTED_LANCE;
    whitePoints +=
        board.inHand.pieceNumber.WhiteLance * PieceValue::IN_HAND_LANCE;
    // Knights
    pieces = board[BB::Type::KNIGHT] & board[BB::Type::ALL_WHITE] &
             ~board[BB::Type::PROMOTED];
    whitePoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                    popcount(pieces[BOTTOM])) *
                   PieceValue::KNIGHT;
    pieces = board[BB::Type::KNIGHT] & board[BB::Type::ALL_WHITE] &
             board[BB::Type::PROMOTED];
    whitePoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                    popcount(pieces[BOTTOM])) *
                   PieceValue::PROMOTED_KNIGHT;
    whitePoints +=
        board.inHand.pieceNumber.WhiteKnight * PieceValue::IN_HAND_KNIGHT;
    // SilverGenerals
    pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_WHITE] &
             ~board[BB::Type::PROMOTED];
    whitePoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                    popcount(pieces[BOTTOM])) *
                   PieceValue::SILVER_GENERAL;
    pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_WHITE] &
             board[BB::Type::PROMOTED];
    whitePoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                    popcount(pieces[BOTTOM])) *
                   PieceValue::PROMOTED_SILVER_GENERAL;
    whitePoints += board.inHand.pieceNumber.WhiteSilverGeneral *
                   PieceValue::IN_HAND_SILVER_GENERAL;
    // GoldGenerals
    pieces = board[BB::Type::GOLD_GENERAL] & board[BB::Type::ALL_WHITE] &
             ~board[BB::Type::PROMOTED];
    whitePoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                    popcount(pieces[BOTTOM])) *
                   PieceValue::GOLD_GENERAL;
    whitePoints += board.inHand.pieceNumber.WhiteGoldGeneral *
                   PieceValue::IN_HAND_GOLD_GENERAL;
    // Bishops
    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
             ~board[BB::Type::PROMOTED];
    whitePoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                    popcount(pieces[BOTTOM])) *
                   PieceValue::BISHOP;
    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
             board[BB::Type::PROMOTED];
    whitePoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                    popcount(pieces[BOTTOM])) *
                   PieceValue::PROMOTED_BISHOP;
    whitePoints +=
        board.inHand.pieceNumber.WhiteBishop * PieceValue::IN_HAND_BISHOP;
    // Rooks
    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
             ~board[BB::Type::PROMOTED];
    whitePoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                    popcount(pieces[BOTTOM])) *
                   PieceValue::ROOK;
    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
             board[BB::Type::PROMOTED];
    whitePoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                    popcount(pieces[BOTTOM])) *
                   PieceValue::PROMOTED_ROOK;
    whitePoints +=
        board.inHand.pieceNumber.WhiteRook * PieceValue::IN_HAND_ROOK;

    // Black
    // Pawns
    pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
             ~board[BB::Type::PROMOTED];
    blackPoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                    popcount(pieces[BOTTOM])) *
                   PieceValue::PAWN;
    pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
             board[BB::Type::PROMOTED];
    blackPoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                    popcount(pieces[BOTTOM])) *
                   PieceValue::PROMOTED_PAWN;
    blackPoints +=
        board.inHand.pieceNumber.BlackPawn * PieceValue::IN_HAND_LANCE;
    // Lances
    pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] &
             ~board[BB::Type::PROMOTED];
    blackPoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                    popcount(pieces[BOTTOM])) *
                   PieceValue::LANCE;
    pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] &
             board[BB::Type::PROMOTED];
    blackPoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                    popcount(pieces[BOTTOM])) *
                   PieceValue::PROMOTED_LANCE;
    blackPoints +=
        board.inHand.pieceNumber.BlackLance * PieceValue::IN_HAND_LANCE;
    // Knights
    pieces = board[BB::Type::KNIGHT] & board[BB::Type::ALL_BLACK] &
             ~board[BB::Type::PROMOTED];
    blackPoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                    popcount(pieces[BOTTOM])) *
                   PieceValue::KNIGHT;
    pieces = board[BB::Type::KNIGHT] & board[BB::Type::ALL_BLACK] &
             board[BB::Type::PROMOTED];
    blackPoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                    popcount(pieces[BOTTOM])) *
                   PieceValue::PROMOTED_KNIGHT;
    blackPoints +=
        board.inHand.pieceNumber.BlackKnight * PieceValue::IN_HAND_KNIGHT;
    // SilverGenerals
    pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_BLACK] &
             ~board[BB::Type::PROMOTED];
    blackPoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                    popcount(pieces[BOTTOM])) *
                   PieceValue::SILVER_GENERAL;
    pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_BLACK] &
             board[BB::Type::PROMOTED];
    blackPoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                    popcount(pieces[BOTTOM])) *
                   PieceValue::PROMOTED_SILVER_GENERAL;
    blackPoints += board.inHand.pieceNumber.BlackSilverGeneral *
                   PieceValue::IN_HAND_SILVER_GENERAL;
    // GoldGenerals
    pieces = board[BB::Type::GOLD_GENERAL] & board[BB::Type::ALL_BLACK] &
             ~board[BB::Type::PROMOTED];
    blackPoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                    popcount(pieces[BOTTOM])) *
                   PieceValue::GOLD_GENERAL;
    blackPoints += board.inHand.pieceNumber.BlackGoldGeneral *
                   PieceValue::IN_HAND_GOLD_GENERAL;
    // Bishops
    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
             ~board[BB::Type::PROMOTED];
    blackPoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                    popcount(pieces[BOTTOM])) *
                   PieceValue::BISHOP;
    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
             board[BB::Type::PROMOTED];
    blackPoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                    popcount(pieces[BOTTOM])) *
                   PieceValue::PROMOTED_BISHOP;
    blackPoints +=
        board.inHand.pieceNumber.BlackBishop * PieceValue::IN_HAND_BISHOP;
    // Rooks
    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
             ~board[BB::Type::PROMOTED];
    blackPoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                    popcount(pieces[BOTTOM])) *
                   PieceValue::ROOK;
    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
             board[BB::Type::PROMOTED];
    blackPoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                    popcount(pieces[BOTTOM])) *
                   PieceValue::PROMOTED_ROOK;
    blackPoints +=
        board.inHand.pieceNumber.BlackRook * PieceValue::IN_HAND_ROOK;

    outValues[index] = whitePoints - blackPoints;
  }
}

void CPU::countWhiteMoves(uint32_t size,
                          int16_t movesPerBoard,
                          const Board& startBoard,
                          Move* inMoves,
                          uint32_t* outOffsets,
                          Bitboard* outValidMoves,
                          Bitboard* outAttackedByEnemy,
                          Bitboard* outPinned,
                          bool* isMate) {
  for (int index = 0; index < size; index++) {
    Board board = startBoard;
    for (int m = 0; m < movesPerBoard; m++) {
      makeMoveWhite(board, inMoves[index * movesPerBoard + m]);
    }

    Bitboard king = board[BB::Type::KING] & board[BB::Type::ALL_WHITE];
    Bitboard notPromoted = ~board[BB::Type::PROMOTED];
    Bitboard checkingPieces, attacked, pieces, moves, slidingChecksPaths,
        attacks, attacksFull, mask, potentialPin, pinned, ourAttacks;
    BitboardIterator iterator;
    Square square;
    size_t numberOfMoves = 0;

    // Non Sliding pieces
    // Pawns
    pieces = board.bbs[BB::Type::PAWN] & board.bbs[BB::Type::ALL_BLACK] &
             notPromoted;
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
    checkingPieces |= (moveSE(king) | moveS(king) | moveSW(king) |
                       moveNE(king) | moveNW(king)) &
                      pieces;
    attacked |= moveNE(pieces) | moveN(pieces) | moveNW(pieces) |
                moveSE(pieces) | moveSW(pieces);
    // Gold generals
    pieces = (board[BB::Type::GOLD_GENERAL] |
              ((board[BB::Type::PAWN] | board[BB::Type::KNIGHT] |
                board[BB::Type::SILVER_GENERAL] | board[BB::Type::LANCE]) &
               board[BB::Type::PROMOTED])) &
             board.bbs[BB::Type::ALL_BLACK];
    checkingPieces |= (moveSE(king) | moveS(king) | moveSW(king) | moveE(king) |
                       moveW(king) | moveN(king)) &
                      pieces;
    attacked |= moveNE(pieces) | moveN(pieces) | moveNW(pieces) |
                moveE(pieces) | moveW(pieces) | moveS(pieces);
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
    attacked |=
        moveNW(pieces) | moveNE(pieces) | moveSE(pieces) | moveSW(pieces);

    // Sliding pieces
    iterator.Init(king);
    iterator.Next();
    Square kingSquare = iterator.GetCurrentSquare();
    Bitboard occupied = board[BB::Type::ALL_BLACK] | board[BB::Type::ALL_WHITE];
    // Lance
    {
      pieces =
          board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] & notPromoted;
      checkingPieces |= CPU::getFileAttacks(kingSquare, occupied) &
                        ~CPU::getRankMask(squareToRank(kingSquare)) & pieces;
      iterator.Init(pieces);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        attacksFull = CPU::getFileAttacks(square, occupied);
        attacked |= attacksFull;
        mask = CPU::getRankMask(squareToRank(square));
        if (!(attacksFull & king & mask)) {
          potentialPin = attacksFull & board[BB::ALL_WHITE];
          attacks = CPU::getFileAttacks(square, occupied & ~potentialPin);
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
      checkingPieces |= (CPU::getRankAttacks(kingSquare, occupied) |
                         CPU::getFileAttacks(kingSquare, occupied)) &
                        pieces;
      iterator.Init(pieces);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        // Check if king is in check without white pieces
        // We have to check all 4 directions
        // left-right
        attacksFull = CPU::getRankAttacks(square, occupied);
        attacked |= attacksFull;
        mask = CPU::getFileMask(squareToFile(square));
        // left
        if (!(attacksFull & king & mask)) {
          potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & mask;
          attacks = CPU::getRankAttacks(square, occupied & ~potentialPin);
          if (attacks & king & mask) {
            pinned |= potentialPin;
          }
        } else {
          slidingChecksPaths |= attacksFull & mask;
        }
        // right
        if (!(attacksFull & king & ~mask)) {
          potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & ~mask;
          attacks = CPU::getRankAttacks(square, occupied & ~potentialPin);
          if (attacks & king & ~mask) {
            pinned |= potentialPin;
          }
        } else {
          slidingChecksPaths |= attacksFull & ~mask;
        }
        // up-down
        attacksFull = CPU::getFileAttacks(square, occupied);
        attacked |= attacksFull;
        mask = CPU::getRankMask(squareToRank(square));
        // up
        if (!(attacksFull & king & mask)) {
          potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & mask;
          attacks = CPU::getFileAttacks(square, occupied & ~potentialPin);
          if (attacks & king & mask) {
            pinned |= potentialPin;
          }
        } else {
          slidingChecksPaths |= attacksFull & mask;
        }
        // down
        if (!(attacksFull & king & ~mask)) {
          potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & ~mask;
          attacks = CPU::getFileAttacks(square, occupied & ~potentialPin);
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
      checkingPieces |= (CPU::getDiagRightAttacks(kingSquare, occupied) |
                         CPU::getDiagLeftAttacks(kingSquare, occupied)) &
                        pieces;
      iterator.Init(pieces);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        // Check if king is in check without white pieces
        // We have to check all 4 directions
        // right diag
        attacksFull = CPU::getDiagRightAttacks(square, occupied);
        attacked |= attacksFull;
        mask = ~CPU::getFileMask(squareToFile(square)) &
               CPU::getRankMask(squareToRank(square));
        // SW
        if (!(attacksFull & king & mask)) {
          potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & mask;
          attacks = CPU::getDiagRightAttacks(square, occupied & ~potentialPin);
          if (attacks & king & mask) {
            pinned |= potentialPin;
          }
        } else {
          slidingChecksPaths |= attacksFull & mask;
        }
        // NE
        if (!(attacksFull & king & ~mask)) {
          potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & ~mask;
          attacks = CPU::getDiagRightAttacks(square, occupied & ~potentialPin);
          if (attacks & king & ~mask) {
            pinned |= potentialPin;
          }
        } else {
          slidingChecksPaths |= attacksFull & ~mask;
        }
        // left diag
        attacksFull = CPU::getDiagLeftAttacks(square, occupied);
        attacked |= attacksFull;
        mask = CPU::getFileMask(squareToFile(square)) &
               CPU::getRankMask(squareToRank(square));
        // NW
        if (!(attacksFull & king & mask)) {
          potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & mask;
          attacks = CPU::getDiagLeftAttacks(square, occupied & ~potentialPin);
          if (attacks & king & mask) {
            pinned |= potentialPin;
          }
        } else {
          slidingChecksPaths |= attacksFull & mask;
        }
        // SE
        if (!(attacksFull & king & ~mask)) {
          potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & ~mask;
          attacks = CPU::getDiagLeftAttacks(square, occupied & ~potentialPin);
          if (attacks & king & ~mask) {
            pinned |= potentialPin;
          }
        } else {
          slidingChecksPaths |= attacksFull & ~mask;
        }
      }
    }

    int numberOfCheckingPieces = popcount(checkingPieces[TOP]) +
                                 popcount(checkingPieces[MID]) +
                                 popcount(checkingPieces[BOTTOM]);

    // King can always move to non attacked squares
    moves = moveNE(king) | moveN(king) | moveNW(king) | moveE(king) |
            moveW(king) | moveSE(king) | moveS(king) | moveSW(king);
    moves &= ~attacked & ~board[BB::Type::ALL_WHITE];
    numberOfMoves +=
        popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);

    Bitboard validMoves;
    // If more then one piece is checking the king and king cannot move its mate
    if (numberOfCheckingPieces > 1) {
      if (numberOfMoves == 0) {
        *isMate = true;
        return;
      }
    } else if (numberOfCheckingPieces == 1) {
      // if king is checked by exactly one piece legal moves can also be block
      // sliding check or capture a checking piece
      validMoves = checkingPieces | (slidingChecksPaths & ~king);
    } else if (numberOfCheckingPieces == 0) {
      // If there is no checks all moves are valid (you cannot capture your own
      // piece)
      validMoves = ~board[BB::Type::ALL_WHITE];
    }

    outValidMoves[index] = validMoves;
    outAttackedByEnemy[index] = attacked;
    outPinned[index] = pinned;

    // Pawn moves
    {
      pieces = ~pinned & board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
               notPromoted;
      moves = moveS(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) +
          popcount(moves[BOTTOM] & ~BOTTOM_RANK) * 2 +  // promotions
          popcount(moves[BOTTOM] & BOTTOM_RANK);        // forced promotion
    }

    // Knight moves
    {
      pieces = ~pinned & board[BB::Type::KNIGHT] & board[BB::Type::ALL_WHITE] &
               notPromoted;
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
      pieces = ~pinned & board[BB::Type::SILVER_GENERAL] &
               board[BB::Type::ALL_WHITE] & notPromoted;
      moves = moveS(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += popcount(moves[TOP]) + popcount(moves[MID]) +
                       popcount(moves[BOTTOM]) * 2;  // promotions
      moves = moveSE(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += popcount(moves[TOP]) + popcount(moves[MID]) +
                       popcount(moves[BOTTOM]) * 2;  // promotions
      moves = moveSW(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += popcount(moves[TOP]) + popcount(moves[MID]) +
                       popcount(moves[BOTTOM]) * 2;  // promotions
      moves = moveNE(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += popcount(moves[TOP]) +
                       popcount(moves[MID] & BOTTOM_RANK) *
                           2 +  // promotion when starting from promotion zone
                       popcount(moves[MID] & ~BOTTOM_RANK) +
                       popcount(moves[BOTTOM]) * 2;  // promotions
      moves = moveNW(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += popcount(moves[TOP]) +
                       popcount(moves[MID] & BOTTOM_RANK) *
                           2 +  // promotion when starting from promotion zone
                       popcount(moves[MID] & ~BOTTOM_RANK) +
                       popcount(moves[BOTTOM]) * 2;  // promotions
    }

    // GoldGenerals moves
    {
      pieces = ~pinned &
               (board[BB::Type::GOLD_GENERAL] |
                ((board[BB::Type::PAWN] | board[BB::Type::LANCE] |
                  board[BB::Type::KNIGHT] | board[BB::Type::SILVER_GENERAL]) &
                 board[BB::Type::PROMOTED])) &
               board[BB::Type::ALL_WHITE];
      moves = moveS(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveSE(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveSW(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveE(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveW(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveN(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
    }

    // Lance moves
    {
      pieces = ~pinned & board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] &
               notPromoted;
      iterator.Init(pieces);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        moves = CPU::getFileAttacks(square, occupied) &
                ~CPU::getRankMask(squareToRank(square)) & validMoves;
        ourAttacks |= moves;
        numberOfMoves +=
            popcount(moves[TOP]) + popcount(moves[MID]) +
            popcount(moves[BOTTOM] & ~BOTTOM_RANK) * 2 +  // promotions
            popcount(moves[BOTTOM] & BOTTOM_RANK);        // forced promotion
      }
    }

    // Bishop moves
    {
      pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
               notPromoted;
      iterator.Init(pieces);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        moves = (CPU::getDiagRightAttacks(square, occupied) |
                 CPU::getDiagLeftAttacks(square, occupied)) &
                validMoves;
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

    // Rook moves
    {
      pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
               notPromoted;
      iterator.Init(pieces);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        moves = (CPU::getRankAttacks(square, occupied) |
                 CPU::getFileAttacks(square, occupied)) &
                validMoves;
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

    // Horse moves
    {
      pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
               board[BB::Type::PROMOTED];
      iterator.Init(pieces);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        Bitboard horse = Bitboard(square);
        moves = (CPU::getDiagRightAttacks(square, occupied) |
                 CPU::getDiagLeftAttacks(square, occupied) | moveN(horse) |
                 moveE(horse) | moveS(horse) | moveW(horse)) &
                validMoves;
        ourAttacks |= moves;
        numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                          popcount(moves[BOTTOM]));
      }
    }

    // Dragon moves
    {
      pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
               board[BB::Type::PROMOTED];
      iterator.Init(pieces);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        Bitboard dragon(square);
        moves = (CPU::getRankAttacks(square, occupied) |
                 CPU::getFileAttacks(square, occupied) | moveNW(dragon) |
                 moveNE(dragon) | moveSE(dragon) | moveSW(dragon)) &
                validMoves;
        ourAttacks |= moves;
        numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                          popcount(moves[BOTTOM]));
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
                notPromoted)) {
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
    if (numberOfMoves == 0) {
      *isMate = true;
      return;
    }
    outOffsets[index] = numberOfMoves;
  }
}

void CPU::countBlackMoves(uint32_t size,
                          int16_t movesPerBoard,
                          const Board& startBoard,
                          Move* inMoves,
                          uint32_t* outOffsets,
                          Bitboard* outValidMoves,
                          Bitboard* outAttackedByEnemy,
                          Bitboard* outPinned,
                          bool* isMate) {
  for (int index = 0; index < size; index++) {
    Board board = startBoard;
    for (int m = 0; m < movesPerBoard; m++) {
      makeMoveWhite(board, inMoves[index * movesPerBoard + m]);
    }

    Bitboard king = board[BB::Type::KING] & board[BB::Type::ALL_BLACK];
    Bitboard notPromoted = ~board[BB::Type::PROMOTED];
    Bitboard checkingPieces, attacked, pieces, moves, slidingChecksPaths,
        attacks, attacksFull, mask, potentialPin, pinned, ourAttacks;
    BitboardIterator iterator;
    Square square;
    size_t numberOfMoves = 0;

    // Non Sliding pieces
    // Pawns
    pieces = board.bbs[BB::Type::PAWN] & board.bbs[BB::Type::ALL_WHITE] &
             notPromoted;
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
    checkingPieces |= (moveNE(king) | moveN(king) | moveNW(king) |
                       moveSE(king) | moveSW(king)) &
                      pieces;
    attacked |= moveSE(pieces) | moveS(pieces) | moveSW(pieces) |
                moveNE(pieces) | moveNW(pieces);
    // gold generals
    pieces = (board[BB::Type::GOLD_GENERAL] |
              ((board[BB::Type::PAWN] | board[BB::Type::KNIGHT] |
                board[BB::Type::SILVER_GENERAL] | board[BB::Type::LANCE]) &
               board[BB::Type::PROMOTED])) &
             board.bbs[BB::Type::ALL_WHITE];
    checkingPieces |= (moveNE(king) | moveN(king) | moveNW(king) | moveE(king) |
                       moveW(king) | moveS(king)) &
                      pieces;
    attacked |= moveSE(pieces) | moveS(pieces) | moveSW(pieces) |
                moveE(pieces) | moveW(pieces) | moveN(pieces);
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
    attacked |=
        moveNW(pieces) | moveNE(pieces) | moveSE(pieces) | moveSW(pieces);

    // Sliding pieces
    iterator.Init(king);
    iterator.Next();
    Square kingSquare = iterator.GetCurrentSquare();
    Bitboard occupied = board[BB::Type::ALL_BLACK] | board[BB::Type::ALL_WHITE];
    // Lance
    {
      pieces =
          board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] & notPromoted;
      checkingPieces |= CPU::getFileAttacks(kingSquare, occupied) &
                        CPU::getRankMask(squareToRank(kingSquare)) & pieces;
      iterator.Init(pieces);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        attacksFull = CPU::getFileAttacks(square, occupied);
        attacked |= attacksFull;
        mask = ~CPU::getRankMask(squareToRank(square));
        if (!(attacksFull & king & mask)) {
          potentialPin = attacksFull & board[BB::ALL_BLACK];
          attacks = CPU::getFileAttacks(square, occupied & ~potentialPin);
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
      checkingPieces |= (CPU::getRankAttacks(kingSquare, occupied) |
                         CPU::getFileAttacks(kingSquare, occupied)) &
                        pieces;
      iterator.Init(pieces);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        // Check if king is in check without white pieces
        // We have to check all 4 directions
        // left-right
        attacksFull = CPU::getRankAttacks(square, occupied);
        attacked |= attacksFull;
        mask = CPU::getFileMask(squareToFile(square));
        // left
        if (!(attacksFull & king & mask)) {
          potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & mask;
          attacks = CPU::getRankAttacks(square, occupied & ~potentialPin);
          if (attacks & king & mask) {
            pinned |= potentialPin;
          }
        } else {
          slidingChecksPaths |= attacksFull & mask;
        }
        // right
        if (!(attacksFull & king & ~mask)) {
          potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & ~mask;
          attacks = CPU::getRankAttacks(square, occupied & ~potentialPin);
          if (attacks & king & ~mask) {
            pinned |= potentialPin;
          }
        } else {
          slidingChecksPaths |= attacksFull & ~mask;
        }
        // up-down
        attacksFull = CPU::getFileAttacks(square, occupied);
        attacked |= attacksFull;
        mask = CPU::getRankMask(squareToRank(square));
        // up
        if (!(attacksFull & king & mask)) {
          potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & mask;
          attacks = CPU::getFileAttacks(square, occupied & ~potentialPin);
          if (attacks & king & mask) {
            pinned |= potentialPin;
          }
        } else {
          slidingChecksPaths |= attacksFull & mask;
        }
        // down
        if (!(attacksFull & king & ~mask)) {
          potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & ~mask;
          attacks = CPU::getFileAttacks(square, occupied & ~potentialPin);
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
      checkingPieces |= (CPU::getDiagRightAttacks(kingSquare, occupied) |
                         CPU::getDiagLeftAttacks(kingSquare, occupied)) &
                        pieces;
      iterator.Init(pieces);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        // Check if king is in check without white pieces
        // We have to check all 4 directions
        // right diag
        attacksFull = CPU::getDiagRightAttacks(square, occupied);
        attacked |= attacksFull;
        mask = ~CPU::getFileMask(squareToFile(square)) &
               CPU::getRankMask(squareToRank(square));
        // SW
        if (!(attacksFull & king & mask)) {
          potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & mask;
          attacks = CPU::getDiagRightAttacks(square, occupied & ~potentialPin);
          if (attacks & king & mask) {
            pinned |= potentialPin;
          }
        } else {
          slidingChecksPaths |= attacksFull & mask;
        }
        // NE
        if (!(attacksFull & king & ~mask)) {
          potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & ~mask;
          attacks = CPU::getDiagRightAttacks(square, occupied & ~potentialPin);
          if (attacks & king & ~mask) {
            pinned |= potentialPin;
          }
        } else {
          slidingChecksPaths |= attacksFull & ~mask;
        }
        // left diag
        attacksFull = CPU::getDiagLeftAttacks(square, occupied);
        attacked |= attacksFull;
        mask = CPU::getFileMask(squareToFile(square)) &
               CPU::getRankMask(squareToRank(square));
        // NW
        if (!(attacksFull & king & mask)) {
          potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & mask;
          attacks = CPU::getDiagLeftAttacks(square, occupied & ~potentialPin);
          if (attacks & king & mask) {
            pinned |= potentialPin;
          }
        } else {
          slidingChecksPaths |= attacksFull & mask;
        }
        // SE
        if (!(attacksFull & king & ~mask)) {
          potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & ~mask;
          attacks = CPU::getDiagLeftAttacks(square, occupied & ~potentialPin);
          if (attacks & king & ~mask) {
            pinned |= potentialPin;
          }
        } else {
          slidingChecksPaths |= attacksFull & ~mask;
        }
      }
    }

    int numberOfCheckingPieces = popcount(checkingPieces[TOP]) +
                                 popcount(checkingPieces[MID]) +
                                 popcount(checkingPieces[BOTTOM]);

    // King can always move to non attacked squares
    moves = moveNE(king) | moveN(king) | moveNW(king) | moveE(king) |
            moveW(king) | moveSE(king) | moveS(king) | moveSW(king);
    moves &= ~attacked & ~board[BB::Type::ALL_BLACK];
    numberOfMoves +=
        popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
    Bitboard validMoves;
    // If more then one piece is checking the king and king cannot move its mate
    if (numberOfCheckingPieces > 1) {
      if (numberOfMoves == 0) {
        *isMate = true;
        return;
      }
    } else if (numberOfCheckingPieces == 1) {
      // if king is checked by exactly one piece legal moves can also be block
      // sliding check or capture a checking piece
      validMoves = checkingPieces | (slidingChecksPaths & ~king);
    } else if (numberOfCheckingPieces == 0) {
      // If there is no checks all moves are valid (you cannot capture your own
      // piece)
      validMoves = ~board[BB::Type::ALL_BLACK];
    }

    outValidMoves[index] = validMoves;
    outAttackedByEnemy[index] = attacked;
    outPinned[index] = pinned;

    // Pawn moves
    {
      pieces = ~pinned & board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
               notPromoted;
      moves = moveN(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += popcount(moves[TOP] & TOP_RANK) +  // forced promotions
                       popcount(moves[TOP] & ~TOP_RANK) * 2 +  // promotions
                       popcount(moves[MID]) + popcount(moves[BOTTOM]);
    }

    // Knight moves
    {
      pieces = ~pinned & board[BB::Type::KNIGHT] & board[BB::Type::ALL_BLACK] &
               notPromoted;
      moves = moveN(moveNE(pieces)) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP] & ~BOTTOM_RANK) +     // forced promotions
          popcount(moves[TOP] & BOTTOM_RANK) * 2 +  // promotions
          popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveN(moveNW(pieces)) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += popcount(moves[TOP] & ~TOP_RANK) +  // forced promotions
                       popcount(moves[TOP] & TOP_RANK) * 2 +  // promotions
                       popcount(moves[MID]) + popcount(moves[BOTTOM]);
    }

    // SilverGenerals moves
    {
      pieces = ~pinned & board[BB::Type::SILVER_GENERAL] &
               board[BB::Type::ALL_BLACK] & notPromoted;
      moves = moveN(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += popcount(moves[TOP]) * 2 +  // promotions
                       popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveNE(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += popcount(moves[TOP]) * 2 +  // promotions
                       popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveNW(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += popcount(moves[TOP]) * 2 +  // promotions
                       popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveSE(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += popcount(moves[TOP]) * 2 +  // promotions
                       popcount(moves[MID] & TOP_RANK) *
                           2 +  // promotion when starting from promotion zone
                       popcount(moves[MID] & ~TOP_RANK) +
                       popcount(moves[BOTTOM]);
      moves = moveSW(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += popcount(moves[TOP]) * 2 +  // promotions
                       popcount(moves[MID] & TOP_RANK) *
                           2 +  // promotion when starting from promotion zone
                       popcount(moves[MID] & ~TOP_RANK) +
                       popcount(moves[BOTTOM]);
    }

    // GoldGenerals moves
    {
      pieces = ~pinned &
               (board[BB::Type::GOLD_GENERAL] |
                ((board[BB::Type::PAWN] | board[BB::Type::LANCE] |
                  board[BB::Type::KNIGHT] | board[BB::Type::SILVER_GENERAL]) &
                 board[BB::Type::PROMOTED])) &
               board[BB::Type::ALL_BLACK];
      moves = moveN(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveNE(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveNW(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveE(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveW(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
      moves = moveS(pieces) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          popcount(moves[TOP]) + popcount(moves[MID]) + popcount(moves[BOTTOM]);
    }

    // Lance moves
    {
      pieces = ~pinned & board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] &
               notPromoted;
      iterator.Init(pieces);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        moves = CPU::getFileAttacks(square, occupied) &
                CPU::getRankMask(squareToRank(square)) & validMoves;
        ourAttacks |= moves;
        numberOfMoves += popcount(moves[TOP] & TOP_RANK) +  // forced promotions
                         popcount(moves[TOP] & ~TOP_RANK) * 2 +  // promotions
                         popcount(moves[MID]) + popcount(moves[BOTTOM]);
      }
    }

    // Bishop moves
    {
      pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
               notPromoted;
      iterator.Init(pieces);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        moves = (CPU::getDiagRightAttacks(square, occupied) |
                 CPU::getDiagLeftAttacks(square, occupied)) &
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
    }

    // Rook moves
    {
      pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
               notPromoted;
      iterator.Init(pieces);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        moves = (CPU::getRankAttacks(square, occupied) |
                 CPU::getFileAttacks(square, occupied)) &
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
    }

    // Horse moves
    {
      pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
               board[BB::Type::PROMOTED];
      iterator.Init(pieces);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        Bitboard horse = Bitboard(square);
        moves = (CPU::getDiagRightAttacks(square, occupied) |
                 CPU::getDiagLeftAttacks(square, occupied) | moveN(horse) |
                 moveE(horse) | moveS(horse) | moveW(horse)) &
                validMoves;
        ourAttacks |= moves;
        numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                          popcount(moves[BOTTOM]));
      }
    }

    // Dragon moves
    {
      pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
               board[BB::Type::PROMOTED];
      iterator.Init(pieces);
      while (iterator.Next()) {
        square = iterator.GetCurrentSquare();
        Bitboard dragon(square);
        moves = (CPU::getRankAttacks(square, occupied) |
                 CPU::getFileAttacks(square, occupied) | moveNW(dragon) |
                 moveNE(dragon) | moveSE(dragon) | moveSW(dragon)) &
                validMoves;
        ourAttacks |= moves;
        numberOfMoves += (popcount(moves[TOP]) + popcount(moves[MID]) +
                          popcount(moves[BOTTOM]));
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
                notPromoted)) {
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
    if (numberOfMoves == 0) {
      *isMate = true;
      return;
    }
    outOffsets[index] = numberOfMoves;
  }
}

void CPU::generateWhiteMoves(uint32_t size,
                             int16_t movesPerBoard,
                             const Board& startBoard,
                             Move* inMoves,
                             uint32_t* inOffsets,
                             Bitboard* inValidMoves,
                             Bitboard* inAttackedByEnemy,
                             Bitboard* inPinned,
                             Move* outMoves) {
  for (int index = 0; index < size; index++) {
    Board board = startBoard;
    uint32_t movesOffset = inOffsets[index] + movesPerBoard * index;
    for (int m = 0; m < movesPerBoard; m++) {
      Move move = inMoves[index * movesPerBoard + m];
      makeMoveWhite(board, move);
      outMoves[movesOffset + m] = move;
    }
    uint32_t moveNumber = movesPerBoard;

    Bitboard validMoves = inValidMoves[index];
    Bitboard notPromoted = ~board[BB::Type::PROMOTED];
    Bitboard pinned = inPinned[index];
    Bitboard pieces, moves, ourAttacks;
    Bitboard occupied = board[BB::Type::ALL_WHITE] | board[BB::Type::ALL_BLACK];
    BitboardIterator movesIterator, iterator;
    Move move;
    uint32_t movesCount = inOffsets[index + 1] - inOffsets[index];
    // Pawn moves
    {
      pieces = ~pinned & board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
               notPromoted;
      moves = moveS(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + N;
        // Promotion
        if (move.to >= WHITE_PROMOTION_START) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
        // Not when forced promotion
        if (move.to < WHITE_PAWN_LANCE_FORECED_PROMOTION_START) {
          move.promotion = 0;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
      }
    }
    // Knight moves
    {
      pieces = ~pinned & board[BB::Type::KNIGHT] & board[BB::Type::ALL_WHITE] &
               notPromoted;
      moves = moveS(moveSE(pieces)) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + N + NW;
        // Promotion
        if (move.to >= WHITE_PROMOTION_START) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
        // Not when forced promotion
        if (move.to < WHITE_HORSE_FORCED_PROMOTION_START) {
          move.promotion = 0;
          outMoves[movesOffset + moveNumber] = move;
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
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
        // Not when forced promotion
        if (move.to < WHITE_HORSE_FORCED_PROMOTION_START) {
          move.promotion = 0;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
      }
    }

    // SilverGenerals moves
    {
      pieces = ~pinned & board[BB::Type::SILVER_GENERAL] &
               board[BB::Type::ALL_WHITE] & notPromoted;
      moves = moveS(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + N;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
        // Promotion
        if (move.to >= WHITE_PROMOTION_START) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
      }
      moves = moveSE(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + NW;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
        // Promotion
        if (move.to >= WHITE_PROMOTION_START) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
      }
      moves = moveSW(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + NE;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
        // Promotion
        if (move.to >= WHITE_PROMOTION_START) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
      }
      moves = moveNE(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + SW;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
        // Promotion
        if (move.to >= WHITE_PROMOTION_START ||
            move.from >= WHITE_PROMOTION_START) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
      }
      moves = moveNW(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + SE;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
        // Promotion
        if (move.to >= WHITE_PROMOTION_START ||
            move.from >= WHITE_PROMOTION_START) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
      }
    }

    // GoldGenerals moves
    {
      pieces = ~pinned &
               (board[BB::Type::GOLD_GENERAL] |
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
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
      moves = moveSE(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + NW;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
      moves = moveSW(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + NE;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
      moves = moveE(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + W;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
      moves = moveW(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + E;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
      moves = moveN(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + S;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }

    // Lances moves
    {
      pieces = ~pinned & board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] &
               notPromoted;
      iterator.Init(pieces);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        moves =
            CPU::getFileAttacks(static_cast<Square>(move.from), occupied) &
            ~CPU::getRankMask(squareToRank(static_cast<Square>(move.from))) &
            validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          // Promotion
          if (move.to >= WHITE_PROMOTION_START) {
            move.promotion = 1;
            outMoves[movesOffset + moveNumber] = move;
            moveNumber++;
          }
          // Not when forced promotion
          if (move.to < WHITE_PAWN_LANCE_FORECED_PROMOTION_START) {
            move.promotion = 0;
            outMoves[movesOffset + moveNumber] = move;
            moveNumber++;
          }
        }
      }
    }

    // Bishop moves
    {
      pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
               notPromoted;
      iterator.Init(pieces);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        moves = (CPU::getDiagRightAttacks(static_cast<Square>(move.from),
                                          occupied) |
                 CPU::getDiagLeftAttacks(static_cast<Square>(move.from),
                                         occupied)) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
          // Promotion
          if (move.to >= WHITE_PROMOTION_START ||
              move.from >= WHITE_PROMOTION_START) {
            move.promotion = 1;
            outMoves[movesOffset + moveNumber] = move;
            moveNumber++;
          }
        }
      }
    }

    // Rook moves
    {
      pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
               notPromoted;
      iterator.Init(pieces);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        moves =
            (CPU::getRankAttacks(static_cast<Square>(move.from), occupied) |
             CPU::getFileAttacks(static_cast<Square>(move.from), occupied)) &
            validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
          // Promotion
          if (move.to >= WHITE_PROMOTION_START ||
              move.from >= WHITE_PROMOTION_START) {
            move.promotion = 1;
            outMoves[movesOffset + moveNumber] = move;
            moveNumber++;
          }
        }
      }
    }
    move.promotion = 0;

    // Horse moves
    {
      pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
               board[BB::Type::PROMOTED];
      iterator.Init(pieces);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        Bitboard horse(static_cast<Square>(move.from));
        moves =
            (CPU::getDiagRightAttacks(static_cast<Square>(move.from),
                                      occupied) |
             CPU::getDiagLeftAttacks(static_cast<Square>(move.from), occupied) |
             moveN(horse) | moveE(horse) | moveS(horse) | moveW(horse)) &
            validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
      }
    }

    // Dragon moves
    {
      pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
               board[BB::Type::PROMOTED];
      iterator.Init(pieces);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        Bitboard dragon(static_cast<Square>(move.from));
        moves = (CPU::getRankAttacks(static_cast<Square>(move.from), occupied) |
                 CPU::getFileAttacks(static_cast<Square>(move.from), occupied) |
                 moveNW(pieces) | moveNE(pieces) | moveSE(pieces) |
                 moveSW(pieces)) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
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
            ~inAttackedByEnemy[index] & ~board[BB::Type::ALL_WHITE];
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
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
                notPromoted)) {
            validFiles |= file;
          }
        }
        legalDropSpots &= validFiles;
        movesIterator.Init(legalDropSpots);
        move.from = WHITE_PAWN_DROP;
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;
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
          outMoves[movesOffset + moveNumber] = move;
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
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
      }
      legalDropSpots = validMoves & ~occupied;
      movesIterator.Init(legalDropSpots);
      while (movesIterator.Next()) {
        if (board.inHand.pieceNumber.WhiteSilverGeneral > 0) {
          move.from = WHITE_SILVER_GENERAL_DROP;
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
        if (board.inHand.pieceNumber.WhiteGoldGeneral > 0) {
          move.from = WHITE_GOLD_GENERAL_DROP;
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
        if (board.inHand.pieceNumber.WhiteBishop > 0) {
          move.from = WHITE_BISHOP_DROP;
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
        if (board.inHand.pieceNumber.WhiteRook > 0) {
          move.from = WHITE_ROOK_DROP;
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
      }
    }

    if (moveNumber > movesCount) {
      std::cout << "Error in:" << std::endl;
      std::cout << boardToSFEN(board) << std::endl;
      std::cout << "More moves generated then counted" << std::endl;
    }
  }
}

void CPU::generateBlackMoves(uint32_t size,
                        int16_t movesPerBoard,
                        const Board& startBoard,
                        Move* inMoves,
                        uint32_t* inOffsets,
                        Bitboard* inValidMoves,
                        Bitboard* inAttackedByEnemy,
                        Bitboard* inPinned,
                        Move* outMoves) {
  for (int index = 0; index < size; index++) {
    Board board = startBoard;
    uint32_t movesOffset = inOffsets[index] + movesPerBoard * index;
    for (int m = 0; m < movesPerBoard; m++) {
      Move move = inMoves[index * movesPerBoard + m];
      makeMoveWhite(board, move);
      outMoves[movesOffset + m] = move;
    }
    uint32_t moveNumber = movesPerBoard;

    Bitboard validMoves = inValidMoves[index];
    Bitboard notPromoted = ~board[BB::Type::PROMOTED];
    Bitboard pinned = inPinned[index];
    Bitboard pieces, moves, ourAttacks;
    Bitboard occupied = board[BB::Type::ALL_WHITE] | board[BB::Type::ALL_BLACK];
    BitboardIterator movesIterator, iterator;
    Move move;
    uint32_t movesCount = inOffsets[index + 1] - inOffsets[index];
    // Pawn moves
    {
      pieces = ~pinned & board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
               notPromoted;
      moves = moveN(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + S;
        // Promotion
        if (move.to <= BLACK_PROMOTION_END) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
        // Not when forced promotion
        if (move.to > BLACK_PAWN_LANCE_FORCE_PROMOTION_END) {
          move.promotion = 0;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
      }
    }
    // Knight moves
    {
      pieces = ~pinned & board[BB::Type::KNIGHT] & board[BB::Type::ALL_BLACK] &
               notPromoted;
      moves = moveN(moveNE(pieces)) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + S + SW;
        // Promotion
        if (move.to <= BLACK_PROMOTION_END) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
        // Not when forced promotion
        if (move.to > BLACK_HORSE_FORCED_PROMOTION_END) {
          move.promotion = 0;
          outMoves[movesOffset + moveNumber] = move;
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
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
        // Not when forced promotion
        if (move.to > BLACK_HORSE_FORCED_PROMOTION_END) {
          move.promotion = 0;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
      }
    }
    // SilverGenerals moves
    {
      pieces = ~pinned & board[BB::Type::SILVER_GENERAL] &
               board[BB::Type::ALL_BLACK] & notPromoted;
      moves = moveN(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + S;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
        // Promotion
        if (move.to <= BLACK_PROMOTION_END) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
      }
      moves = moveNE(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + SW;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
        // Promotion
        if (move.to <= BLACK_PROMOTION_END) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
      }
      moves = moveNW(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + SE;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
        // Promotion
        if (move.to <= BLACK_PROMOTION_END) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
      }
      moves = moveSE(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + NW;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
        // Promotion
        if (move.to <= BLACK_PROMOTION_END ||
            move.from <= BLACK_PROMOTION_END) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
      }
      moves = moveSW(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + NE;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
        // Promotion
        if (move.to <= BLACK_PROMOTION_END ||
            move.from <= BLACK_PROMOTION_END) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
      }
    }
    // GoldGenerals moves
    {
      pieces = ~pinned &
               (board[BB::Type::GOLD_GENERAL] |
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
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
      moves = moveNE(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + SW;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
      moves = moveNW(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + SE;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
      moves = moveE(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + W;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
      moves = moveW(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + E;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
      moves = moveS(pieces) & validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.from = move.to + N;
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
    // Lances moves
    {
      pieces = ~pinned & board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] &
               notPromoted;
      iterator.Init(pieces);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        moves = CPU::getFileAttacks(static_cast<Square>(move.from), occupied) &
                CPU::getRankMask(squareToRank(static_cast<Square>(move.from))) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          // Promotion
          if (move.to <= BLACK_PROMOTION_END) {
            move.promotion = 1;
            outMoves[movesOffset + moveNumber] = move;
            moveNumber++;
          }
          // Not when forced promotion
          if (move.to > BLACK_PAWN_LANCE_FORCE_PROMOTION_END) {
            move.promotion = 0;
            outMoves[movesOffset + moveNumber] = move;
            moveNumber++;
          }
        }
      }
    }
    // Bishop moves
    {
      pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
               notPromoted;
      iterator.Init(pieces);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        moves = (CPU::getDiagRightAttacks(static_cast<Square>(move.from),
                                          occupied) |
                 CPU::getDiagLeftAttacks(static_cast<Square>(move.from),
                                         occupied)) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
          // Promotion
          if (move.to <= BLACK_PROMOTION_END ||
              move.from <= BLACK_PROMOTION_END) {
            move.promotion = 1;
            outMoves[movesOffset + moveNumber] = move;
            moveNumber++;
          }
        }
      }
    }
    // Rook moves
    {
      pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
               notPromoted;
      iterator.Init(pieces);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        moves = (CPU::getRankAttacks(static_cast<Square>(move.from), occupied) |
             CPU::getFileAttacks(static_cast<Square>(move.from), occupied)) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          move.promotion = 0;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
          // Promotion
          if (move.to <= BLACK_PROMOTION_END ||
              move.from <= BLACK_PROMOTION_END) {
            move.promotion = 1;
            outMoves[movesOffset + moveNumber] = move;
            moveNumber++;
          }
        }
      }
    }
    move.promotion = 0;
    // Horse moves
    {
      pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
               board[BB::Type::PROMOTED];
      iterator.Init(pieces);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        Bitboard horse(static_cast<Square>(move.from));
        moves = (CPU::getDiagRightAttacks(static_cast<Square>(move.from),
                                          occupied) |
             CPU::getDiagLeftAttacks(static_cast<Square>(move.from), occupied) |
                 moveN(horse) | moveE(horse) | moveS(horse) | moveW(horse)) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
      }
    }
    // Dragon moves
    {
      pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
               board[BB::Type::PROMOTED];
      iterator.Init(pieces);
      while (iterator.Next()) {
        move.from = iterator.GetCurrentSquare();
        Bitboard dragon(static_cast<Square>(move.from));
        moves = (CPU::getRankAttacks(static_cast<Square>(move.from), occupied) |
                 CPU::getFileAttacks(static_cast<Square>(move.from), occupied) |
                 moveNW(dragon) | moveNE(dragon) | moveSE(dragon) |
                 moveSW(dragon)) &
                validMoves;
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
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
            ~inAttackedByEnemy[index] & ~board[BB::Type::ALL_BLACK];
        ourAttacks |= moves;
        movesIterator.Init(moves);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
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
                notPromoted)) {
            validFiles |= file;
          }
        }
        legalDropSpots &= validFiles;
        move.from = BLACK_PAWN_DROP;
        movesIterator.Init(legalDropSpots);
        while (movesIterator.Next()) {
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;
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
          outMoves[movesOffset + moveNumber] = move;
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
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
      }
      legalDropSpots = validMoves & ~occupied;
      movesIterator.Init(legalDropSpots);
      while (movesIterator.Next()) {
        if (board.inHand.pieceNumber.BlackSilverGeneral > 0) {
          move.from = BLACK_SILVER_GENERAL_DROP;
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
        if (board.inHand.pieceNumber.BlackGoldGeneral > 0) {
          move.from = BLACK_GOLD_GENERAL_DROP;
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
        if (board.inHand.pieceNumber.BlackBishop > 0) {
          move.from = BLACK_BISHOP_DROP;
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
        if (board.inHand.pieceNumber.BlackRook > 0) {
          move.from = BLACK_ROOK_DROP;
          move.to = movesIterator.GetCurrentSquare();
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
      }
    }

    if (moveNumber > movesCount) {
      std::cout << "Error in:" << std::endl;
      std::cout << boardToSFEN(board) << std::endl;
      std::cout << "More moves generated then counted" << std::endl;
    }
  }
}

void CPU::gatherValuesMin(uint32_t size,
                          int16_t movesPerBoard,
                          uint32_t* inOffsets,
                          int16_t* inValues,
                          int16_t* outValues) {
  for (int index = 0; index < size; index++) {
    int16_t minValue = INT16_MAX;
    for (int i = inOffsets[index]; i < inOffsets[index + 1]; i++) {
      minValue = std::min(minValue, inValues[i]);
    }
    outValues[index] = minValue;
  }
}

void CPU::gatherValuesMax(uint32_t size,
                          int16_t movesPerBoard,
                          uint32_t* inOffsets,
                          int16_t* inValues,
                          int16_t* outValues) {
  for (int index = 0; index < size; index++) {
    int16_t maxValue = INT16_MIN;
    for (int i = inOffsets[index]; i < inOffsets[index + 1]; i++) {
      maxValue = std::max(maxValue, inValues[i]);
    }
    outValues[index] = maxValue;
  }
}
}  // namespace engine
}  // namespace shogi