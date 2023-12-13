//#include "MoveGen.h"
//namespace shogi {
//namespace engine {
//size_t countWhiteMoves(const Board& board,
//                       Bitboard& outValidMoves,
//                       Bitboard& attackedByEnemy) {
//  Bitboard king = board[BB::Type::KING] & board[BB::Type::ALL_WHITE];
//  Bitboard notPromoted = ~board[BB::Type::PROMOTED];
//  Bitboard checkingPieces, attacked, pieces, moves, slidingChecksPaths, attacks,
//      attacksFull, mask, potentialPin, pinned, ourAttacks;
//  BitboardIterator iterator;
//  Square square;
//  size_t numberOfMoves = 0;
//
//  // Non Sliding pieces
//  // Pawns
//  pieces =
//      board.bbs[BB::Type::PAWN] & board.bbs[BB::Type::ALL_BLACK] & notPromoted;
//  checkingPieces |= moveS(king) & pieces;
//  attacked |= moveN(pieces);
//  // Knights
//  pieces = board.bbs[BB::Type::KNIGHT] & board.bbs[BB::Type::ALL_BLACK] &
//           notPromoted;
//  checkingPieces |= moveS(moveSE(king) | moveSW(king)) & pieces;
//  attacked |= moveN(moveNE(pieces) | moveNW(pieces));
//  // Silve generals
//  pieces = board.bbs[BB::Type::SILVER_GENERAL] &
//           board.bbs[BB::Type::ALL_BLACK] & notPromoted;
//  checkingPieces |= (moveSE(king) | moveS(king) | moveSW(king) | moveNE(king) |
//                     moveNW(king)) &
//                    pieces;
//  attacked |= moveNE(pieces) | moveN(pieces) | moveNW(pieces) | moveSE(pieces) |
//              moveSW(pieces);
//  // Gold generals
//  pieces = (board[BB::Type::GOLD_GENERAL] |
//            ((board[BB::Type::PAWN] | board[BB::Type::KNIGHT] |
//              board[BB::Type::SILVER_GENERAL] | board[BB::Type::LANCE]) &
//             board[BB::Type::PROMOTED])) &
//           board.bbs[BB::Type::ALL_BLACK];
//  checkingPieces |= (moveSE(king) | moveS(king) | moveSW(king) | moveE(king) |
//                     moveW(king) | moveN(king)) &
//                    pieces;
//  attacked |= moveNE(pieces) | moveN(pieces) | moveNW(pieces) | moveE(pieces) |
//              moveW(pieces) | moveS(pieces);
//  // Horse (non sliding part)
//  pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
//           board[BB::Type::PROMOTED];
//  checkingPieces |=
//      (moveN(king) | moveE(king) | moveS(king) | moveW(king)) & pieces;
//  attacked |= moveN(pieces) | moveE(pieces) | moveS(pieces) | moveW(pieces);
//  // Dragon (non sliding part)
//  pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
//           board[BB::Type::PROMOTED];
//  checkingPieces |=
//      (moveNW(king) | moveNE(king) | moveSE(king) | moveSW(king)) & pieces;
//  attacked |= moveNW(pieces) | moveNE(pieces) | moveSE(pieces) | moveSW(pieces);
//
//  // Sliding pieces
//  iterator.Init(king);
//  iterator.Next();
//  Square kingSquare = iterator.GetCurrentSquare();
//  Bitboard occupied = board[BB::Type::ALL_BLACK] | board[BB::Type::ALL_WHITE];
//  // Lance
//  {
//    pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] & notPromoted;
//    checkingPieces |= getFileAttacks(kingSquare, occupied) &
//                      ~getRankMask(squareToRank(kingSquare)) & pieces;
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      square = iterator.GetCurrentSquare();
//      attacksFull = getFileAttacks(square, occupied);
//      attacked |= attacksFull;
//      mask = getRankMask(squareToRank(square));
//      if (!(attacksFull & king & mask)) {
//        potentialPin = attacksFull & board[BB::ALL_WHITE];
//        attacks = getFileAttacks(square, occupied & ~potentialPin);
//        if (attacks & king & mask) {
//          pinned |= potentialPin;
//        }
//      } else {
//        slidingChecksPaths |= attacksFull & mask;
//      }
//    }
//  }
//
//  // Rook and dragon
//  {
//    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK];
//    checkingPieces |= (getRankAttacks(kingSquare, occupied) |
//                       getFileAttacks(kingSquare, occupied)) &
//                      pieces;
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      square = iterator.GetCurrentSquare();
//      // Check if king is in check without white pieces
//      // We have to check all 4 directions
//      // left-right
//      attacksFull = getRankAttacks(square, occupied);
//      attacked |= attacksFull;
//      mask = getFileMask(squareToFile(square));
//      // left
//      if (!(attacksFull & king & mask)) {
//        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & mask;
//        attacks = getRankAttacks(square, occupied & ~potentialPin);
//        if (attacks & king & mask) {
//          pinned |= potentialPin;
//        }
//      } else {
//        slidingChecksPaths |= attacksFull & mask;
//      }
//      // right
//      if (!(attacksFull & king & ~mask)) {
//        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & ~mask;
//        attacks = getRankAttacks(square, occupied & ~potentialPin);
//        if (attacks & king & ~mask) {
//          pinned |= potentialPin;
//        }
//      } else {
//        slidingChecksPaths |= attacksFull & ~mask;
//      }
//      // up-down
//      attacksFull = getFileAttacks(square, occupied);
//      attacked |= attacksFull;
//      mask = getRankMask(squareToRank(square));
//      // up
//      if (!(attacksFull & king & mask)) {
//        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & mask;
//        attacks = getFileAttacks(square, occupied & ~potentialPin);
//        if (attacks & king & mask) {
//          pinned |= potentialPin;
//        }
//      } else {
//        slidingChecksPaths |= attacksFull & mask;
//      }
//      // down
//      if (!(attacksFull & king & ~mask)) {
//        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & ~mask;
//        attacks = getFileAttacks(square, occupied & ~potentialPin);
//        if (attacks & king & ~mask) {
//          pinned |= potentialPin;
//        }
//      } else {
//        slidingChecksPaths |= attacksFull & ~mask;
//      }
//    }
//  }
//
//  // Bishop and horse pins
//  {
//    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK];
//    checkingPieces |= (getDiagRightAttacks(kingSquare, occupied) |
//                       getDiagLeftAttacks(kingSquare, occupied)) &
//                      pieces;
//    iterator.Init(board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK]);
//    while (iterator.Next()) {
//      square = iterator.GetCurrentSquare();
//      // Check if king is in check without white pieces
//      // We have to check all 4 directions
//      // right diag
//      attacksFull = getDiagRightAttacks(square, occupied);
//      attacked |= attacksFull;
//      mask = ~getFileMask(squareToFile(square)) &
//             getRankMask(squareToRank(square));
//      // SW
//      if (!(attacksFull & king & mask)) {
//        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & mask;
//        attacks = getDiagRightAttacks(square, occupied & ~potentialPin);
//        if (attacks & king & mask) {
//          pinned |= potentialPin;
//        }
//      } else {
//        slidingChecksPaths |= attacksFull & mask;
//      }
//      // NE
//      if (!(attacksFull & king & ~mask)) {
//        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & ~mask;
//        attacks = getDiagRightAttacks(square, occupied & ~potentialPin);
//        if (attacks & king & ~mask) {
//          pinned |= potentialPin;
//        }
//      } else {
//        slidingChecksPaths |= attacksFull & ~mask;
//      }
//      // left diag
//      attacksFull = getDiagLeftAttacks(square, occupied);
//      attacked |= attacksFull;
//      mask =
//          getFileMask(squareToFile(square)) & getRankMask(squareToRank(square));
//      // NW
//      if (!(attacksFull & king & mask)) {
//        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & mask;
//        attacks = getDiagLeftAttacks(square, occupied & ~potentialPin);
//        if (attacks & king & mask) {
//          pinned |= potentialPin;
//        }
//      } else {
//        slidingChecksPaths |= attacksFull & mask;
//      }
//      // SE
//      if (!(attacksFull & king & ~mask)) {
//        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & ~mask;
//        attacks = getDiagLeftAttacks(square, occupied & ~potentialPin);
//        if (attacks & king & ~mask) {
//          pinned |= potentialPin;
//        }
//      } else {
//        slidingChecksPaths |= attacksFull & ~mask;
//      }
//    }
//  }
//
//  int numberOfCheckingPieces = std::popcount<uint32_t>(checkingPieces[TOP]) +
//                               std::popcount<uint32_t>(checkingPieces[MID]) +
//                               std::popcount<uint32_t>(checkingPieces[BOTTOM]);
//
//  // King can always move to non attacked squares
//  moves = moveNE(king) | moveN(king) | moveNW(king) | moveE(king) |
//          moveW(king) | moveSE(king) | moveS(king) | moveSW(king);
//  moves &= ~attacked & ~board[BB::Type::ALL_WHITE];
//  numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
//                   std::popcount<uint32_t>(moves[MID]) +
//                   std::popcount<uint32_t>(moves[BOTTOM]);
//
//  Bitboard validMoves;
//  if (numberOfCheckingPieces == 1) {
//    // if king is checked by exactly one piece legal moves can also be block
//    // sliding check or capture a checking piece
//    validMoves = checkingPieces | (slidingChecksPaths & ~king);
//  } else if (numberOfCheckingPieces == 0) {
//    // If there is no checks all moves are valid (you cannot capture your own
//    // piece)
//    validMoves = ~board[BB::Type::ALL_WHITE];
//  }
//
//  outValidMoves = validMoves;
//  attackedByEnemy = attacked;
//
//  // Pawn moves
//  {
//    pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] & notPromoted;
//    moves = moveS(pieces) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
//                     std::popcount<uint32_t>(moves[MID]) +
//                     std::popcount<uint32_t>(moves[BOTTOM] & BOTTOM_RANK) *
//                         2 +  // promotions
//                     std::popcount<uint32_t>(moves[BOTTOM] &
//                                             ~BOTTOM_RANK);  // forced promotion
//  }
//
//  // Knight moves
//  {
//    pieces = board[BB::Type::KNIGHT] & board[BB::Type::ALL_WHITE] & notPromoted;
//    moves = moveS(moveSE(pieces)) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves +=
//        std::popcount<uint32_t>(moves[TOP]) +
//        std::popcount<uint32_t>(moves[MID]) +
//        std::popcount<uint32_t>(moves[BOTTOM] & TOP_RANK) * 2 +  // promotions
//        std::popcount<uint32_t>(moves[BOTTOM] & ~TOP_RANK);  // forced promotion
//    moves = moveS(moveSW(pieces)) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves +=
//        std::popcount<uint32_t>(moves[TOP]) +
//        std::popcount<uint32_t>(moves[MID]) +
//        std::popcount<uint32_t>(moves[BOTTOM] & TOP_RANK) * 2 +  // promotions
//        std::popcount<uint32_t>(moves[BOTTOM] & ~TOP_RANK);  // forced promotion
//  }
//
//  // SilverGenerals moves
//  {
//    pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_WHITE] &
//             notPromoted;
//    moves = moveS(pieces) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
//                     std::popcount<uint32_t>(moves[MID]) +
//                     std::popcount<uint32_t>(moves[BOTTOM]) * 2;  // promotions
//    moves = moveSE(pieces) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
//                     std::popcount<uint32_t>(moves[MID]) +
//                     std::popcount<uint32_t>(moves[BOTTOM]) * 2;  // promotions
//    moves = moveSW(pieces) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
//                     std::popcount<uint32_t>(moves[MID]) +
//                     std::popcount<uint32_t>(moves[BOTTOM]) * 2;  // promotions
//    moves = moveNE(pieces) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
//                     std::popcount<uint32_t>(moves[MID] & BOTTOM_RANK) *
//                         2 +  // promotion when starting from promotion zone
//                     std::popcount<uint32_t>(moves[MID] & ~BOTTOM_RANK) +
//                     std::popcount<uint32_t>(moves[BOTTOM]) * 2;  // promotions
//    moves = moveNW(pieces) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
//                     std::popcount<uint32_t>(moves[MID] & BOTTOM_RANK) *
//                         2 +  // promotion when starting from promotion zone
//                     std::popcount<uint32_t>(moves[MID] & ~BOTTOM_RANK) +
//                     std::popcount<uint32_t>(moves[BOTTOM]) * 2;  // promotions
//  }
//
//  // GoldGenerals moves
//  {
//    pieces = (board[BB::Type::GOLD_GENERAL] |
//              ((board[BB::Type::PAWN] | board[BB::Type::LANCE] |
//                board[BB::Type::KNIGHT] | board[BB::Type::SILVER_GENERAL]) &
//               board[BB::Type::PROMOTED])) &
//             board[BB::Type::ALL_WHITE];
//    moves = moveS(pieces) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
//                     std::popcount<uint32_t>(moves[MID]) +
//                     std::popcount<uint32_t>(moves[BOTTOM]);
//    moves = moveSE(pieces) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
//                     std::popcount<uint32_t>(moves[MID]) +
//                     std::popcount<uint32_t>(moves[BOTTOM]);
//    moves = moveSW(pieces) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
//                     std::popcount<uint32_t>(moves[MID]) +
//                     std::popcount<uint32_t>(moves[BOTTOM]);
//    moves = moveE(pieces) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
//                     std::popcount<uint32_t>(moves[MID]) +
//                     std::popcount<uint32_t>(moves[BOTTOM]);
//    moves = moveW(pieces) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
//                     std::popcount<uint32_t>(moves[MID]) +
//                     std::popcount<uint32_t>(moves[BOTTOM]);
//    moves = moveN(pieces) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
//                     std::popcount<uint32_t>(moves[MID]) +
//                     std::popcount<uint32_t>(moves[BOTTOM]);
//  }
//
//  // Lance moves
//  {
//    pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] & notPromoted;
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      square = iterator.GetCurrentSquare();
//      moves = getFileAttacks(square, occupied) &
//              ~getRankMask(squareToRank(square)) & validMoves;
//      ourAttacks |= moves;
//      numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
//                       std::popcount<uint32_t>(moves[MID]) +
//                       std::popcount<uint32_t>(moves[BOTTOM] & BOTTOM_RANK) *
//                           2 +  // promotions
//                       std::popcount<uint32_t>(
//                           moves[BOTTOM] & ~BOTTOM_RANK);  // forced promotion
//    }
//  }
//
//  // Bishop moves
//  {
//    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] & notPromoted;
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      square = iterator.GetCurrentSquare();
//      moves = (getDiagRightAttacks(square, occupied) |
//               getDiagLeftAttacks(square, occupied)) &
//              validMoves;
//      ourAttacks |= moves;
//      if (square > WHITE_PROMOTION_START) {  // Starting from promotion zone
//        numberOfMoves += (std::popcount<uint32_t>(moves[TOP]) +
//                          std::popcount<uint32_t>(moves[MID]) +
//                          std::popcount<uint32_t>(moves[BOTTOM])) *
//                         2;
//      } else {
//        numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
//                         std::popcount<uint32_t>(moves[MID]) +
//                         std::popcount<uint32_t>(moves[BOTTOM]) *
//                             2;  // end in promotion Zone
//      }
//    }
//  }
//
//  // Rook moves
//  {
//    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] & notPromoted;
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      square = iterator.GetCurrentSquare();
//      moves = (getRankAttacks(square, occupied) |
//               getFileAttacks(square, occupied)) &
//              validMoves;
//      ourAttacks |= moves;
//      if (square > WHITE_PROMOTION_START) {  // Starting from promotion zone
//        numberOfMoves += (std::popcount<uint32_t>(moves[TOP]) +
//                          std::popcount<uint32_t>(moves[MID]) +
//                          std::popcount<uint32_t>(moves[BOTTOM])) *
//                         2;
//      } else {
//        numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
//                         std::popcount<uint32_t>(moves[MID]) +
//                         std::popcount<uint32_t>(moves[BOTTOM]) *
//                             2;  // end in promotion Zone
//      }
//    }
//  }
//
//  // Horse moves
//  {
//    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
//             board[BB::Type::PROMOTED];
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      square = iterator.GetCurrentSquare();
//      Bitboard horse = Bitboard(square);
//      moves = (getDiagRightAttacks(square, occupied) |
//               getDiagLeftAttacks(square, occupied) | moveN(horse) |
//               moveE(horse) | moveS(horse) | moveW(horse)) &
//              validMoves;
//      ourAttacks |= moves;
//      numberOfMoves += (std::popcount<uint32_t>(moves[TOP]) +
//                        std::popcount<uint32_t>(moves[MID]) +
//                        std::popcount<uint32_t>(moves[BOTTOM]));
//    }
//  }
//
//  // Dragon moves
//  {
//    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
//             board[BB::Type::PROMOTED];
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      square = iterator.GetCurrentSquare();
//      Bitboard dragon(square);
//      moves =
//          (getRankAttacks(square, occupied) | getFileAttacks(square, occupied) |
//           moveNW(dragon) | moveNE(dragon) | moveSE(dragon) | moveSW(dragon)) &
//          validMoves;
//      ourAttacks |= moves;
//      numberOfMoves += (std::popcount<uint32_t>(moves[TOP]) +
//                        std::popcount<uint32_t>(moves[MID]) +
//                        std::popcount<uint32_t>(moves[BOTTOM]));
//    }
//  }
//
//  // Drop moves
//  {
//    Bitboard legalDropSpots;
//    // Pawns
//    if (board.inHand.pieceNumber.WhitePawn > 0) {
//      legalDropSpots = ~occupied;
//      // Cannot drop on last rank
//      legalDropSpots[BOTTOM] &= ~BOTTOM_RANK;
//      // Cannot drop to give checkmate
//      pieces = board[BB::Type::KING] & board[BB::Type::ALL_BLACK];
//      // All valid enemy king moves
//      moves =
//          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
//           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
//          ourAttacks;
//      // If there is only one spot pawn cannot block it
//      if (std::popcount<uint32_t>(moves[TOP]) +
//              std::popcount<uint32_t>(moves[MID]) +
//              std::popcount<uint32_t>(moves[BOTTOM]) ==
//          1) {
//        legalDropSpots &= ~moveN(moves);
//      }
//      // Cannot drop on file with other pawn
//      Bitboard validFiles;
//      for (int fileIdx = 0; fileIdx < BOARD_DIM; fileIdx++) {
//        Bitboard file = getFullFile(fileIdx);
//        if (!(file & board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
//              notPromoted)) {
//          validFiles |= file;
//        }
//      }
//      legalDropSpots &= validFiles;
//      numberOfMoves += std::popcount<uint32_t>(legalDropSpots[TOP]) +
//                       std::popcount<uint32_t>(legalDropSpots[MID]) +
//                       std::popcount<uint32_t>(legalDropSpots[BOTTOM]);
//    }
//    if (board.inHand.pieceNumber.WhiteLance > 0) {
//      legalDropSpots = ~occupied;
//      // Cannot drop on last rank
//      legalDropSpots[BOTTOM] &= ~BOTTOM_RANK;
//      numberOfMoves += std::popcount<uint32_t>(legalDropSpots[TOP]) +
//                       std::popcount<uint32_t>(legalDropSpots[MID]) +
//                       std::popcount<uint32_t>(legalDropSpots[BOTTOM]);
//    }
//    if (board.inHand.pieceNumber.WhiteKnight > 0) {
//      legalDropSpots = ~occupied;
//      // Cannot drop on last two ranks
//      legalDropSpots[BOTTOM] &= TOP_RANK;
//      numberOfMoves += std::popcount<uint32_t>(legalDropSpots[TOP]) +
//                       std::popcount<uint32_t>(legalDropSpots[MID]) +
//                       std::popcount<uint32_t>(legalDropSpots[BOTTOM]);
//    }
//    legalDropSpots = ~occupied;
//    numberOfMoves += ((board.inHand.pieceNumber.WhiteSilverGeneral > 0) +
//                      (board.inHand.pieceNumber.WhiteGoldGeneral > 0) +
//                      (board.inHand.pieceNumber.WhiteBishop > 0) +
//                      (board.inHand.pieceNumber.WhiteRook > 0)) *
//                     (std::popcount<uint32_t>(legalDropSpots[TOP]) +
//                      std::popcount<uint32_t>(legalDropSpots[MID]) +
//                      std::popcount<uint32_t>(legalDropSpots[BOTTOM]));
//  }
//  return numberOfMoves;
//}
//
//size_t countBlackMoves(const Board& board,
//                       Bitboard& outValidMoves,
//                       Bitboard& attackedByEnemy) {
//  Bitboard king = board[BB::Type::KING] & board[BB::Type::ALL_BLACK];
//  Bitboard notPromoted = ~board[BB::Type::PROMOTED];
//  Bitboard checkingPieces, attacked, pieces, moves, slidingChecksPaths, attacks,
//      attacksFull, mask, potentialPin, pinned, ourAttacks;
//  BitboardIterator iterator;
//  Square square;
//  size_t numberOfMoves = 0;
//
//  // Non Sliding pieces
//  // Pawns
//  pieces =
//      board.bbs[BB::Type::PAWN] & board.bbs[BB::Type::ALL_WHITE] & notPromoted;
//  checkingPieces |= moveN(king) & pieces;
//  attacked |= moveS(pieces);
//  // Knights
//  pieces = board.bbs[BB::Type::KNIGHT] & board.bbs[BB::Type::ALL_WHITE] &
//           notPromoted;
//  checkingPieces |= moveN(moveNE(king) | moveNW(king)) & pieces;
//  attacked |= moveS(moveSE(pieces) | moveSW(pieces));
//  // Silver generals
//  pieces = board.bbs[BB::Type::SILVER_GENERAL] &
//           board.bbs[BB::Type::ALL_WHITE] & notPromoted;
//  checkingPieces |= (moveNE(king) | moveN(king) | moveNW(king) | moveSE(king) |
//                     moveSW(king)) &
//                    pieces;
//  attacked |= moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveNE(pieces) |
//              moveNW(pieces);
//  // gold generals
//  pieces = (board[BB::Type::GOLD_GENERAL] |
//            ((board[BB::Type::PAWN] | board[BB::Type::KNIGHT] |
//              board[BB::Type::SILVER_GENERAL] | board[BB::Type::LANCE]) &
//             board[BB::Type::PROMOTED])) &
//           board.bbs[BB::Type::ALL_WHITE];
//  checkingPieces |= (moveNE(king) | moveN(king) | moveNW(king) | moveE(king) |
//                     moveW(king) | moveS(king)) &
//                    pieces;
//  attacked |= moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveE(pieces) |
//              moveW(pieces) | moveN(pieces);
//  // Horse (non sliding part)
//  pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
//           board[BB::Type::PROMOTED];
//  checkingPieces |=
//      (moveN(king) | moveE(king) | moveS(king) | moveW(king)) & pieces;
//  attacked |= moveN(pieces) | moveE(pieces) | moveS(pieces) | moveW(pieces);
//  // Dragon (non sldiing part)
//  pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
//           board[BB::Type::PROMOTED];
//  checkingPieces |=
//      (moveNW(king) | moveNE(king) | moveSE(king) | moveSW(king)) & pieces;
//  attacked |= moveNW(pieces) | moveNE(pieces) | moveSE(pieces) | moveSW(pieces);
//
//  // Sliding pieces
//  iterator.Init(king);
//  iterator.Next();
//  Square kingSquare = iterator.GetCurrentSquare();
//  Bitboard occupied = board[BB::Type::ALL_BLACK] | board[BB::Type::ALL_WHITE];
//  // Lance
//  {
//    pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] & notPromoted;
//    checkingPieces |= getFileAttacks(kingSquare, occupied) &
//                      getRankMask(squareToRank(kingSquare)) & pieces;
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      square = iterator.GetCurrentSquare();
//      attacksFull = getFileAttacks(square, occupied);
//      attacked |= attacksFull;
//      mask = ~getRankMask(squareToRank(square));
//      if (!(attacksFull & king & mask)) {
//        potentialPin = attacksFull & board[BB::ALL_BLACK];
//        attacks = getFileAttacks(square, occupied & ~potentialPin);
//        if (attacks & king & mask) {
//          pinned |= potentialPin;
//        }
//      } else {
//        slidingChecksPaths |= attacksFull & mask;
//      }
//    }
//  }
//
//  // Rook and dragon
//  {
//    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE];
//    checkingPieces |= (getRankAttacks(kingSquare, occupied) |
//                       getFileAttacks(kingSquare, occupied)) &
//                      pieces;
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      square = iterator.GetCurrentSquare();
//      // Check if king is in check without white pieces
//      // We have to check all 4 directions
//      // left-right
//      attacksFull = getRankAttacks(square, occupied);
//      attacked |= attacksFull;
//      mask = getFileMask(squareToFile(square));
//      // left
//      if (!(attacksFull & king & mask)) {
//        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & mask;
//        attacks = getRankAttacks(square, occupied & ~potentialPin);
//        if (attacks & king & mask) {
//          pinned |= potentialPin;
//        }
//      } else {
//        slidingChecksPaths |= attacksFull & mask;
//      }
//      // right
//      if (!(attacksFull & king & ~mask)) {
//        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & ~mask;
//        attacks = getRankAttacks(square, occupied & ~potentialPin);
//        if (attacks & king & ~mask) {
//          pinned |= potentialPin;
//        }
//      } else {
//        slidingChecksPaths |= attacksFull & ~mask;
//      }
//      // up-down
//      attacksFull = getFileAttacks(square, occupied);
//      attacked |= attacksFull;
//      mask = getRankMask(squareToRank(square));
//      // up
//      if (!(attacksFull & king & mask)) {
//        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & mask;
//        attacks = getFileAttacks(square, occupied & ~potentialPin);
//        if (attacks & king & mask) {
//          pinned |= potentialPin;
//        }
//      } else {
//        slidingChecksPaths |= attacksFull & mask;
//      }
//      // down
//      if (!(attacksFull & king & ~mask)) {
//        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & ~mask;
//        attacks = getFileAttacks(square, occupied & ~potentialPin);
//        if (attacks & king & ~mask) {
//          pinned |= potentialPin;
//        }
//      } else {
//        slidingChecksPaths |= attacksFull & ~mask;
//      }
//    }
//  }
//
//  // Bishop and horse pins
//  {
//    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE];
//    checkingPieces |= (getDiagRightAttacks(kingSquare, occupied) |
//                       getDiagLeftAttacks(kingSquare, occupied)) &
//                      pieces;
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      square = iterator.GetCurrentSquare();
//      // Check if king is in check without white pieces
//      // We have to check all 4 directions
//      // right diag
//      attacksFull = getDiagRightAttacks(square, occupied);
//      attacked |= attacksFull;
//      mask = ~getFileMask(squareToFile(square)) &
//             getRankMask(squareToRank(square));
//      // SW
//      if (!(attacksFull & king & mask)) {
//        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & mask;
//        attacks = getDiagRightAttacks(square, occupied & ~potentialPin);
//        if (attacks & king & mask) {
//          pinned |= potentialPin;
//        }
//      } else {
//        slidingChecksPaths |= attacksFull & mask;
//      }
//      // NE
//      if (!(attacksFull & king & ~mask)) {
//        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & ~mask;
//        attacks = getDiagRightAttacks(square, occupied & ~potentialPin);
//        if (attacks & king & ~mask) {
//          pinned |= potentialPin;
//        }
//      } else {
//        slidingChecksPaths |= attacksFull & ~mask;
//      }
//      // left diag
//      attacksFull = getDiagLeftAttacks(square, occupied);
//      attacked |= attacksFull;
//      mask =
//          getFileMask(squareToFile(square)) & getRankMask(squareToRank(square));
//      // NW
//      if (!(attacksFull & king & mask)) {
//        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & mask;
//        attacks = getDiagLeftAttacks(square, occupied & ~potentialPin);
//        if (attacks & king & mask) {
//          pinned |= potentialPin;
//        }
//      } else {
//        slidingChecksPaths |= attacksFull & mask;
//      }
//      // SE
//      if (!(attacksFull & king & ~mask)) {
//        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & ~mask;
//        attacks = getDiagLeftAttacks(square, occupied & ~potentialPin);
//        if (attacks & king & ~mask) {
//          pinned |= potentialPin;
//        }
//      } else {
//        slidingChecksPaths |= attacksFull & ~mask;
//      }
//    }
//  }
//
//  int numberOfCheckingPieces = std::popcount<uint32_t>(checkingPieces[TOP]) +
//                               std::popcount<uint32_t>(checkingPieces[MID]) +
//                               std::popcount<uint32_t>(checkingPieces[BOTTOM]);
//
//  // King can always move to non attacked squares
//  moves = moveNE(king) | moveN(king) | moveNW(king) | moveE(king) |
//          moveW(king) | moveSE(king) | moveS(king) | moveSW(king);
//  moves &= ~attacked & ~board[BB::Type::ALL_BLACK];
//  numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
//                   std::popcount<uint32_t>(moves[MID]) +
//                   std::popcount<uint32_t>(moves[BOTTOM]);
//  Bitboard validMoves;
//  if (numberOfCheckingPieces == 1) {
//    // if king is checked by exactly one piece legal moves can also be block
//    // sliding check or capture a checking piece
//    validMoves = checkingPieces | (slidingChecksPaths & ~king);
//  } else if (numberOfCheckingPieces == 0) {
//    // If there is no checks all moves are valid (you cannot capture your own
//    // piece)
//    validMoves = ~board[BB::Type::ALL_BLACK];
//  }
//
//  outValidMoves = validMoves;
//  attackedByEnemy = attacked;
//
//  // Pawn moves
//  {
//    pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] & notPromoted;
//    moves = moveN(pieces) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves +=
//        std::popcount<uint32_t>(moves[TOP] & TOP_RANK) +  // forced promotions
//        std::popcount<uint32_t>(moves[TOP] & ~TOP_RANK) * 2 +  // promotions
//        std::popcount<uint32_t>(moves[MID]) +
//        std::popcount<uint32_t>(moves[BOTTOM]);
//  }
//
//  // Knight moves
//  {
//    pieces = board[BB::Type::KNIGHT] & board[BB::Type::ALL_BLACK] & notPromoted;
//    moves = moveN(moveNE(pieces)) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves +=
//        std::popcount<uint32_t>(moves[TOP] & TOP_RANK) +  // forced promotions
//        std::popcount<uint32_t>(moves[TOP] & ~TOP_RANK) * 2 +  // promotions
//        std::popcount<uint32_t>(moves[MID]) +
//        std::popcount<uint32_t>(moves[BOTTOM]);
//    moves = moveN(moveNW(pieces)) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves +=
//        std::popcount<uint32_t>(moves[TOP] & TOP_RANK) +  // forced promotions
//        std::popcount<uint32_t>(moves[TOP] & ~TOP_RANK) * 2 +  // promotions
//        std::popcount<uint32_t>(moves[MID]) +
//        std::popcount<uint32_t>(moves[BOTTOM]);
//  }
//
//  // SilverGenerals moves
//  {
//    pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_BLACK] &
//             notPromoted;
//    moves = moveN(pieces) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) * 2 +  // promotions
//                     std::popcount<uint32_t>(moves[MID]) +
//                     std::popcount<uint32_t>(moves[BOTTOM]);
//    moves = moveNE(pieces) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) * 2 +  // promotions
//                     std::popcount<uint32_t>(moves[MID]) +
//                     std::popcount<uint32_t>(moves[BOTTOM]);
//    moves = moveNW(pieces) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) * 2 +  // promotions
//                     std::popcount<uint32_t>(moves[MID]) +
//                     std::popcount<uint32_t>(moves[BOTTOM]);
//    moves = moveSE(pieces) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) * 2 +  // promotions
//                     std::popcount<uint32_t>(moves[MID] & TOP_RANK) *
//                         2 +  // promotion when starting from promotion zone
//                     std::popcount<uint32_t>(moves[MID] & ~TOP_RANK) +
//                     std::popcount<uint32_t>(moves[BOTTOM]);
//    moves = moveSW(pieces) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) * 2 +  // promotions
//                     std::popcount<uint32_t>(moves[MID] & TOP_RANK) *
//                         2 +  // promotion when starting from promotion zone
//                     std::popcount<uint32_t>(moves[MID] & ~TOP_RANK) +
//                     std::popcount<uint32_t>(moves[BOTTOM]);
//  }
//
//  // GoldGenerals moves
//  {
//    pieces = (board[BB::Type::GOLD_GENERAL] |
//              ((board[BB::Type::PAWN] | board[BB::Type::LANCE] |
//                board[BB::Type::KNIGHT] | board[BB::Type::SILVER_GENERAL]) &
//               board[BB::Type::PROMOTED])) &
//             board[BB::Type::ALL_BLACK];
//    moves = moveN(pieces) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
//                     std::popcount<uint32_t>(moves[MID]) +
//                     std::popcount<uint32_t>(moves[BOTTOM]);
//    moves = moveNE(pieces) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
//                     std::popcount<uint32_t>(moves[MID]) +
//                     std::popcount<uint32_t>(moves[BOTTOM]);
//    moves = moveNW(pieces) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
//                     std::popcount<uint32_t>(moves[MID]) +
//                     std::popcount<uint32_t>(moves[BOTTOM]);
//    moves = moveE(pieces) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
//                     std::popcount<uint32_t>(moves[MID]) +
//                     std::popcount<uint32_t>(moves[BOTTOM]);
//    moves = moveW(pieces) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
//                     std::popcount<uint32_t>(moves[MID]) +
//                     std::popcount<uint32_t>(moves[BOTTOM]);
//    moves = moveS(pieces) & validMoves;
//    ourAttacks |= moves;
//    numberOfMoves += std::popcount<uint32_t>(moves[TOP]) +
//                     std::popcount<uint32_t>(moves[MID]) +
//                     std::popcount<uint32_t>(moves[BOTTOM]);
//  }
//
//  // Lance moves
//  {
//    pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] & notPromoted;
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      square = iterator.GetCurrentSquare();
//      moves = getFileAttacks(square, occupied) &
//              getRankMask(squareToRank(square)) & validMoves;
//      ourAttacks |= moves;
//      numberOfMoves +=
//          std::popcount<uint32_t>(moves[TOP] & TOP_RANK) +  // forced promotions
//          std::popcount<uint32_t>(moves[TOP] & ~TOP_RANK) * 2 +  // promotions
//          std::popcount<uint32_t>(moves[MID]) +
//          std::popcount<uint32_t>(moves[BOTTOM]);
//    }
//  }
//
//  // Bishop moves
//  {
//    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] & notPromoted;
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      square = iterator.GetCurrentSquare();
//      moves = (getDiagRightAttacks(square, occupied) |
//               getDiagLeftAttacks(square, occupied)) &
//              validMoves;
//      ourAttacks |= moves;
//      if (square < BLACK_PROMOTION_END) {  // Starting from promotion zone
//        numberOfMoves += (std::popcount<uint32_t>(moves[TOP]) +
//                          std::popcount<uint32_t>(moves[MID]) +
//                          std::popcount<uint32_t>(moves[BOTTOM])) *
//                         2;
//      } else {
//        numberOfMoves +=
//            std::popcount<uint32_t>(moves[TOP]) * 2 +  // end in promotion Zone
//            std::popcount<uint32_t>(moves[MID]) +
//            std::popcount<uint32_t>(moves[BOTTOM]);
//      }
//    }
//  }
//
//  // Rook moves
//  {
//    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] & notPromoted;
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      square = iterator.GetCurrentSquare();
//      moves = (getRankAttacks(square, occupied) |
//               getFileAttacks(square, occupied)) &
//              validMoves;
//      ourAttacks |= moves;
//      if (square < BLACK_PROMOTION_END) {  // Starting from promotion zone
//        numberOfMoves += (std::popcount<uint32_t>(moves[TOP]) +
//                          std::popcount<uint32_t>(moves[MID]) +
//                          std::popcount<uint32_t>(moves[BOTTOM])) *
//                         2;
//      } else {
//        numberOfMoves +=
//            std::popcount<uint32_t>(moves[TOP]) * 2 +  // end in promotion Zone
//            std::popcount<uint32_t>(moves[MID]) +
//            std::popcount<uint32_t>(moves[BOTTOM]);
//      }
//    }
//  }
//
//  // Horse moves
//  {
//    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
//             board[BB::Type::PROMOTED];
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      square = iterator.GetCurrentSquare();
//      Bitboard horse = Bitboard(square);
//      moves = (getDiagRightAttacks(square, occupied) |
//               getDiagLeftAttacks(square, occupied) | moveN(horse) |
//               moveE(horse) | moveS(horse) | moveW(horse)) &
//              validMoves;
//      ourAttacks |= moves;
//      numberOfMoves += (std::popcount<uint32_t>(moves[TOP]) +
//                        std::popcount<uint32_t>(moves[MID]) +
//                        std::popcount<uint32_t>(moves[BOTTOM]));
//    }
//  }
//
//  // Dragon moves
//  {
//    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
//             board[BB::Type::PROMOTED];
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      square = iterator.GetCurrentSquare();
//      Bitboard dragon(square);
//      moves =
//          (getRankAttacks(square, occupied) | getFileAttacks(square, occupied) |
//           moveNW(dragon) | moveNE(dragon) | moveSE(dragon) | moveSW(dragon)) &
//          validMoves;
//      ourAttacks |= moves;
//      numberOfMoves += (std::popcount<uint32_t>(moves[TOP]) +
//                        std::popcount<uint32_t>(moves[MID]) +
//                        std::popcount<uint32_t>(moves[BOTTOM]));
//    }
//  }
//
//  // Drop moves
//  {
//    Bitboard legalDropSpots;
//    // Pawns
//    if (board.inHand.pieceNumber.BlackPawn > 0) {
//      legalDropSpots = ~occupied;
//      // Cannot drop on last rank
//      legalDropSpots[TOP] &= ~TOP_RANK;
//      // Cannot drop to give checkmate
//      pieces = board[BB::Type::KING] & board[BB::Type::ALL_WHITE];
//      // All valid enemy king moves
//      moves =
//          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
//           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
//          ourAttacks;
//      // If there is only one spot pawn cannot block it
//      if (std::popcount<uint32_t>(moves[TOP]) +
//              std::popcount<uint32_t>(moves[MID]) +
//              std::popcount<uint32_t>(moves[BOTTOM]) ==
//          1) {
//        legalDropSpots &= ~moveS(moves);
//      }
//      // Cannot drop on file with other pawn
//      Bitboard validFiles;
//      for (int fileIdx = 0; fileIdx < BOARD_DIM; fileIdx++) {
//        Bitboard file = getFullFile(fileIdx);
//        if (!(file & board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
//              notPromoted)) {
//          validFiles |= file;
//        }
//      }
//      legalDropSpots &= validFiles;
//      numberOfMoves += std::popcount<uint32_t>(legalDropSpots[TOP]) +
//                       std::popcount<uint32_t>(legalDropSpots[MID]) +
//                       std::popcount<uint32_t>(legalDropSpots[BOTTOM]);
//    }
//    if (board.inHand.pieceNumber.BlackLance > 0) {
//      legalDropSpots = ~occupied;
//      // Cannot drop on last rank
//      legalDropSpots[TOP] &= ~TOP_RANK;
//      numberOfMoves += std::popcount<uint32_t>(legalDropSpots[TOP]) +
//                       std::popcount<uint32_t>(legalDropSpots[MID]) +
//                       std::popcount<uint32_t>(legalDropSpots[BOTTOM]);
//    }
//    if (board.inHand.pieceNumber.BlackKnight > 0) {
//      legalDropSpots = ~occupied;
//      // Cannot drop on last two ranks
//      legalDropSpots[TOP] &= TOP_RANK;
//      numberOfMoves += std::popcount<uint32_t>(legalDropSpots[TOP]) +
//                       std::popcount<uint32_t>(legalDropSpots[MID]) +
//                       std::popcount<uint32_t>(legalDropSpots[BOTTOM]);
//    }
//    legalDropSpots = ~occupied;
//    numberOfMoves += ((board.inHand.pieceNumber.BlackSilverGeneral > 0) +
//                      (board.inHand.pieceNumber.BlackGoldGeneral > 0) +
//                      (board.inHand.pieceNumber.BlackBishop > 0) +
//                      (board.inHand.pieceNumber.BlackRook > 0)) *
//                     (std::popcount<uint32_t>(legalDropSpots[TOP]) +
//                      std::popcount<uint32_t>(legalDropSpots[MID]) +
//                      std::popcount<uint32_t>(legalDropSpots[BOTTOM]));
//  }
//
//  return numberOfMoves;
//}
//
//void generateWhiteMoves(const Board& board,
//                        const Bitboard& validMoves,
//                        const Bitboard& attackedByEnemy,
//                        Move* movesArray,
//                        size_t offset) {
//  Bitboard notPromoted = ~board[BB::Type::PROMOTED];
//  Bitboard pieces, moves, ourAttacks;
//  Bitboard occupied = board[BB::Type::ALL_WHITE] | board[BB::Type::ALL_BLACK];
//  BitboardIterator movesIterator, iterator;
//  Move move;
//  Move* currentMove = movesArray + offset;
//  // Pawn moves
//  {
//    pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] & notPromoted;
//    moves = moveS(pieces) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + N;
//      // Promotion
//      if (move.to >= WHITE_PROMOTION_START) {
//        move.promotion = 1;
//        *currentMove = move;
//        currentMove++;
//      }
//      // Not when forced promotion
//      if (move.to < WHITE_PAWN_LANCE_FORECED_PROMOTION_START) {
//        move.promotion = 0;
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//  }
//  // Knight moves
//  {
//    pieces = board[BB::Type::KNIGHT] & board[BB::Type::ALL_WHITE] & notPromoted;
//    moves = moveS(moveSE(pieces)) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + N + NW;
//      // Promotion
//      if (move.to >= WHITE_PROMOTION_START) {
//        move.promotion = 1;
//        *currentMove = move;
//        currentMove++;
//      }
//      // Not when forced promotion
//      if (move.to < WHITE_HORSE_FORCED_PROMOTION_START) {
//        move.promotion = 0;
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//    moves = moveS(moveSW(pieces)) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + N + NE;
//      // Promotion
//      if (move.to >= WHITE_PROMOTION_START) {
//        move.promotion = 1;
//        *currentMove = move;
//        currentMove++;
//      }
//      // Not when forced promotion
//      if (move.to < WHITE_HORSE_FORCED_PROMOTION_START) {
//        move.promotion = 0;
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//  }
//
//  // SilverGenerals moves
//  {
//    pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_WHITE] &
//             notPromoted;
//    moves = moveS(pieces) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + N;
//      move.promotion = 0;
//      *currentMove = move;
//      currentMove++;
//      // Promotion
//      if (move.to >= WHITE_PROMOTION_START) {
//        move.promotion = 1;
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//    moves = moveSE(pieces) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + NW;
//      move.promotion = 0;
//      *currentMove = move;
//      currentMove++;
//      // Promotion
//      if (move.to >= WHITE_PROMOTION_START) {
//        move.promotion = 1;
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//    moves = moveSW(pieces) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + NE;
//      move.promotion = 0;
//      *currentMove = move;
//      currentMove++;
//      // Promotion
//      if (move.to >= WHITE_PROMOTION_START) {
//        move.promotion = 1;
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//    moves = moveNE(pieces) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + SW;
//      move.promotion = 0;
//      *currentMove = move;
//      currentMove++;
//      // Promotion
//      if (move.to >= WHITE_PROMOTION_START ||
//          move.from >= WHITE_PROMOTION_START) {
//        move.promotion = 1;
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//    moves = moveNW(pieces) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + SE;
//      move.promotion = 0;
//      *currentMove = move;
//      currentMove++;
//      // Promotion
//      if (move.to >= WHITE_PROMOTION_START ||
//          move.from >= WHITE_PROMOTION_START) {
//        move.promotion = 1;
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//  }
//
//  // GoldGenerals moves
//  {
//    pieces = (board[BB::Type::GOLD_GENERAL] |
//              ((board[BB::Type::PAWN] | board[BB::Type::LANCE] |
//                board[BB::Type::KNIGHT] | board[BB::Type::SILVER_GENERAL]) &
//               board[BB::Type::PROMOTED])) &
//             board[BB::Type::ALL_WHITE];
//    moves = moveS(pieces) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + N;
//      move.promotion = 0;
//      *currentMove = move;
//      currentMove++;
//    }
//    moves = moveSE(pieces) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + NW;
//      move.promotion = 0;
//      *currentMove = move;
//      currentMove++;
//    }
//    moves = moveSW(pieces) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + NE;
//      move.promotion = 0;
//      *currentMove = move;
//      currentMove++;
//    }
//    moves = moveE(pieces) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + W;
//      move.promotion = 0;
//      *currentMove = move;
//      currentMove++;
//    }
//    moves = moveW(pieces) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + E;
//      move.promotion = 0;
//      *currentMove = move;
//      currentMove++;
//    }
//    moves = moveN(pieces) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + S;
//      move.promotion = 0;
//      *currentMove = move;
//      currentMove++;
//    }
//  }
//
//  // Lances moves
//  {
//    pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] & notPromoted;
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      move.from = iterator.GetCurrentSquare();
//      moves = getFileAttacks(static_cast<Square>(move.from), occupied) &
//              ~getRankMask(squareToRank(static_cast<Square>(move.from))) &
//              validMoves;
//      ourAttacks |= moves;
//      movesIterator.Init(moves);
//      while (movesIterator.Next()) {
//        move.to = movesIterator.GetCurrentSquare();
//        // Promotion
//        if (move.to >= WHITE_PROMOTION_START) {
//          move.promotion = 1;
//          *currentMove = move;
//          currentMove++;
//        }
//        // Not when forced promotion
//        if (move.to < WHITE_PAWN_LANCE_FORECED_PROMOTION_START) {
//          move.promotion = 0;
//          *currentMove = move;
//          currentMove++;
//        }
//      }
//    }
//  }
//
//  // Bishop moves
//  {
//    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] & notPromoted;
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      move.from = iterator.GetCurrentSquare();
//      moves = (getDiagRightAttacks(static_cast<Square>(move.from), occupied) |
//               getDiagLeftAttacks(static_cast<Square>(move.from), occupied)) &
//              validMoves;
//      ourAttacks |= moves;
//      movesIterator.Init(moves);
//      while (movesIterator.Next()) {
//        move.to = movesIterator.GetCurrentSquare();
//        move.promotion = 0;
//        *currentMove = move;
//        currentMove++;
//        // Promotion
//        if (move.to >= WHITE_PROMOTION_START ||
//            move.from >= WHITE_PROMOTION_START) {
//          move.promotion = 1;
//          *currentMove = move;
//          currentMove++;
//        }
//      }
//    }
//  }
//
//  // Rook moves
//  {
//    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] & notPromoted;
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      move.from = iterator.GetCurrentSquare();
//      moves = (getRankAttacks(static_cast<Square>(move.from), occupied) |
//               getFileAttacks(static_cast<Square>(move.from), occupied)) &
//              validMoves;
//      ourAttacks |= moves;
//      movesIterator.Init(moves);
//      while (movesIterator.Next()) {
//        move.to = movesIterator.GetCurrentSquare();
//        move.promotion = 0;
//        *currentMove = move;
//        currentMove++;
//        // Promotion
//        if (move.to >= WHITE_PROMOTION_START ||
//            move.from >= WHITE_PROMOTION_START) {
//          move.promotion = 1;
//          *currentMove = move;
//          currentMove++;
//        }
//      }
//    }
//  }
//  move.promotion = 0;
//
//  // Horse moves
//  {
//    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
//             board[BB::Type::PROMOTED];
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      move.from = iterator.GetCurrentSquare();
//      moves = (getDiagRightAttacks(static_cast<Square>(move.from), occupied) |
//               getDiagLeftAttacks(static_cast<Square>(move.from), occupied) |
//               moveN(pieces) | moveE(pieces) | moveS(pieces) | moveW(pieces)) &
//              validMoves;
//      ourAttacks |= moves;
//      movesIterator.Init(moves);
//      while (movesIterator.Next()) {
//        move.to = movesIterator.GetCurrentSquare();
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//  }
//
//  // Dragon moves
//  {
//    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
//             board[BB::Type::PROMOTED];
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      move.from = iterator.GetCurrentSquare();
//      moves =
//          (getRankAttacks(static_cast<Square>(move.from), occupied) |
//           getFileAttacks(static_cast<Square>(move.from), occupied) |
//           moveNW(pieces) | moveNE(pieces) | moveSE(pieces) | moveSW(pieces)) &
//          validMoves;
//      ourAttacks |= moves;
//      movesIterator.Init(moves);
//      while (movesIterator.Next()) {
//        move.to = movesIterator.GetCurrentSquare();
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//  }
//
//  // King moves
//  {
//    pieces = board[BB::Type::KING] & board[BB::Type::ALL_WHITE];
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      move.from = iterator.GetCurrentSquare();
//      moves =
//          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
//           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
//          ~attackedByEnemy & ~board[BB::Type::ALL_WHITE];
//      ourAttacks |= moves;
//      movesIterator.Init(moves);
//      while (movesIterator.Next()) {
//        move.to = movesIterator.GetCurrentSquare();
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//  }
//
//  // Generate Drop moves
//  {
//    Bitboard legalDropSpots;
//    // Pawns
//    if (board.inHand.pieceNumber.WhitePawn > 0) {
//      legalDropSpots = ~occupied;
//      // Cannot drop on last rank
//      legalDropSpots[BOTTOM] &= ~BOTTOM_RANK;
//      // Cannot drop to give checkmate
//      pieces = board[BB::Type::KING] & board[BB::Type::ALL_BLACK];
//      // All valid enemy king moves
//      moves =
//          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
//           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
//          ~ourAttacks & ~board[BB::Type::ALL_BLACK];
//      // If there is only one spot pawn cannot block it
//      if (std::popcount<uint32_t>(moves[TOP]) +
//              std::popcount<uint32_t>(moves[MID]) +
//              std::popcount<uint32_t>(moves[BOTTOM]) ==
//          1) {
//        legalDropSpots &= ~moveN(moves);
//      }
//      // Cannot drop on file with other pawn
//      Bitboard validFiles;
//      for (int fileIdx = 0; fileIdx < BOARD_DIM; fileIdx++) {
//        Bitboard file = getFullFile(fileIdx);
//        if (!(file & board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
//              notPromoted)) {
//          validFiles |= file;
//        }
//      }
//      legalDropSpots &= validFiles;
//      movesIterator.Init(legalDropSpots);
//      move.from = WHITE_PAWN_DROP;
//      while (movesIterator.Next()) {
//        move.to = movesIterator.GetCurrentSquare();
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//    if (board.inHand.pieceNumber.WhiteLance > 0) {
//      legalDropSpots = ~occupied;
//      // Cannot drop on last rank
//      legalDropSpots[BOTTOM] &= ~BOTTOM_RANK;
//      movesIterator.Init(legalDropSpots);
//      move.from = WHITE_KNIGHT_DROP;
//      while (movesIterator.Next()) {
//        move.to = movesIterator.GetCurrentSquare();
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//    if (board.inHand.pieceNumber.WhiteKnight > 0) {
//      legalDropSpots = ~occupied;
//      // Cannot drop on last two ranks
//      legalDropSpots[BOTTOM] &= TOP_RANK;
//      movesIterator.Init(legalDropSpots);
//      move.from = WHITE_LANCE_DROP;
//      while (movesIterator.Next()) {
//        move.to = movesIterator.GetCurrentSquare();
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//    legalDropSpots = ~occupied;
//    movesIterator.Init(legalDropSpots);
//    while (movesIterator.Next()) {
//      if (board.inHand.pieceNumber.WhiteSilverGeneral > 0) {
//        move.from = WHITE_SILVER_GENERAL_DROP;
//        move.to = movesIterator.GetCurrentSquare();
//        *currentMove = move;
//        currentMove++;
//      }
//      if (board.inHand.pieceNumber.WhiteGoldGeneral > 0) {
//        move.from = WHITE_GOLD_GENERAL_DROP;
//        move.to = movesIterator.GetCurrentSquare();
//        *currentMove = move;
//        currentMove++;
//      }
//      if (board.inHand.pieceNumber.WhiteBishop > 0) {
//        move.from = WHITE_BISHOP_DROP;
//        move.to = movesIterator.GetCurrentSquare();
//        *currentMove = move;
//        currentMove++;
//      }
//      if (board.inHand.pieceNumber.WhiteRook > 0) {
//        move.from = WHITE_ROOK_DROP;
//        move.to = movesIterator.GetCurrentSquare();
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//  }
//}
//
//void generateBlackMoves(const Board& board,
//                        const Bitboard& validMoves,
//                        const Bitboard& attackedByEnemy,
//                        Move* movesArray,
//                        size_t offset) {
//  Bitboard notPromoted = ~board[BB::Type::PROMOTED];
//  Bitboard pieces, moves, ourAttacks;
//  Bitboard occupied = board[BB::Type::ALL_WHITE] | board[BB::Type::ALL_BLACK];
//  BitboardIterator movesIterator, iterator;
//  Move move;
//  Move* currentMove = movesArray + offset;
//  // Pawn moves
//  {
//    pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] & notPromoted;
//    moves = moveN(pieces) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + S;
//      // Promotion
//      if (move.to <= BLACK_PROMOTION_END) {
//        move.promotion = 1;
//        *currentMove = move;
//        currentMove++;
//      }
//      // Not when forced promotion
//      if (move.to > BLACK_PAWN_LANCE_FORCE_PROMOTION_END) {
//        move.promotion = 0;
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//  }
//  // Knight moves
//  {
//    pieces = board[BB::Type::KNIGHT] & board[BB::Type::ALL_BLACK] & notPromoted;
//    moves = moveN(moveNE(pieces)) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + S + SW;
//      // Promotion
//      if (move.to <= BLACK_PROMOTION_END) {
//        move.promotion = 1;
//        *currentMove = move;
//        currentMove++;
//      }
//      // Not when forced promotion
//      if (move.to > BLACK_HORSE_FORCED_PROMOTION_END) {
//        move.promotion = 0;
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//    moves = moveN(moveNW(pieces)) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + S + SE;
//      // Promotion
//      if (move.to <= BLACK_PROMOTION_END) {
//        move.promotion = 1;
//        *currentMove = move;
//        currentMove++;
//      }
//      // Not when forced promotion
//      if (move.to > BLACK_HORSE_FORCED_PROMOTION_END) {
//        move.promotion = 0;
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//  }
//  // SilverGenerals moves
//  {
//    pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_BLACK] &
//             notPromoted;
//    moves = moveN(pieces) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + S;
//      move.promotion = 0;
//      *currentMove = move;
//      currentMove++;
//      // Promotion
//      if (move.to <= BLACK_PROMOTION_END) {
//        move.promotion = 1;
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//    moves = moveNE(pieces) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + SW;
//      move.promotion = 0;
//      *currentMove = move;
//      currentMove++;
//      // Promotion
//      if (move.to <= BLACK_PROMOTION_END) {
//        move.promotion = 1;
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//    moves = moveNW(pieces) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + SE;
//      move.promotion = 0;
//      *currentMove = move;
//      currentMove++;
//      // Promotion
//      if (move.to <= BLACK_PROMOTION_END) {
//        move.promotion = 1;
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//    moves = moveSE(pieces) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + NW;
//      move.promotion = 0;
//      *currentMove = move;
//      currentMove++;
//      // Promotion
//      if (move.to <= BLACK_PROMOTION_END || move.from <= BLACK_PROMOTION_END) {
//        move.promotion = 1;
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//    moves = moveSW(pieces) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + NE;
//      move.promotion = 0;
//      *currentMove = move;
//      currentMove++;
//      // Promotion
//      if (move.to <= BLACK_PROMOTION_END || move.from <= BLACK_PROMOTION_END) {
//        move.promotion = 1;
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//  }
//  // GoldGenerals moves
//  {
//    pieces = (board[BB::Type::GOLD_GENERAL] |
//              ((board[BB::Type::PAWN] | board[BB::Type::LANCE] |
//                board[BB::Type::KNIGHT] | board[BB::Type::SILVER_GENERAL]) &
//               board[BB::Type::PROMOTED])) &
//             board[BB::Type::ALL_BLACK];
//    moves = moveN(pieces) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + S;
//      move.promotion = 0;
//      *currentMove = move;
//      currentMove++;
//    }
//    moves = moveNE(pieces) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + SW;
//      move.promotion = 0;
//      *currentMove = move;
//      currentMove++;
//    }
//    moves = moveNW(pieces) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + SE;
//      move.promotion = 0;
//      *currentMove = move;
//      currentMove++;
//    }
//    moves = moveE(pieces) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + W;
//      move.promotion = 0;
//      *currentMove = move;
//      currentMove++;
//    }
//    moves = moveW(pieces) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + E;
//      move.promotion = 0;
//      *currentMove = move;
//      currentMove++;
//    }
//    moves = moveS(pieces) & validMoves;
//    ourAttacks |= moves;
//    movesIterator.Init(moves);
//    while (movesIterator.Next()) {
//      move.to = movesIterator.GetCurrentSquare();
//      move.from = move.to + N;
//      move.promotion = 0;
//      *currentMove = move;
//      currentMove++;
//    }
//  }
//  // Lances moves
//  {
//    pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] & notPromoted;
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      move.from = iterator.GetCurrentSquare();
//      moves = getFileAttacks(static_cast<Square>(move.from), occupied) &
//              getRankMask(squareToRank(static_cast<Square>(move.from))) &
//              validMoves;
//      ourAttacks |= moves;
//      movesIterator.Init(moves);
//      while (movesIterator.Next()) {
//        move.to = movesIterator.GetCurrentSquare();
//        // Promotion
//        if (move.to <= BLACK_PROMOTION_END) {
//          move.promotion = 1;
//          *currentMove = move;
//          currentMove++;
//        }
//        // Not when forced promotion
//        if (move.to > BLACK_PAWN_LANCE_FORCE_PROMOTION_END) {
//          move.promotion = 0;
//          *currentMove = move;
//          currentMove++;
//        }
//      }
//    }
//  }
//  // Bishop moves
//  {
//    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] & notPromoted;
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      move.from = iterator.GetCurrentSquare();
//      moves = (getDiagRightAttacks(static_cast<Square>(move.from), occupied) |
//               getDiagLeftAttacks(static_cast<Square>(move.from), occupied)) &
//              validMoves;
//      ourAttacks |= moves;
//      movesIterator.Init(moves);
//      while (movesIterator.Next()) {
//        move.to = movesIterator.GetCurrentSquare();
//        move.promotion = 0;
//        *currentMove = move;
//        currentMove++;
//        // Promotion
//        if (move.to <= BLACK_PROMOTION_END ||
//            move.from <= BLACK_PROMOTION_END) {
//          move.promotion = 1;
//          *currentMove = move;
//          currentMove++;
//        }
//      }
//    }
//  }
//  // Rook moves
//  {
//    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] & notPromoted;
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      move.from = iterator.GetCurrentSquare();
//      moves = (getRankAttacks(static_cast<Square>(move.from), occupied) |
//               getFileAttacks(static_cast<Square>(move.from), occupied)) &
//              validMoves;
//      ourAttacks |= moves;
//      movesIterator.Init(moves);
//      while (movesIterator.Next()) {
//        move.to = movesIterator.GetCurrentSquare();
//        move.promotion = 0;
//        *currentMove = move;
//        currentMove++;
//        // Promotion
//        if (move.to <= BLACK_PROMOTION_END ||
//            move.from <= BLACK_PROMOTION_END) {
//          move.promotion = 1;
//          *currentMove = move;
//          currentMove++;
//        }
//      }
//    }
//  }
//  move.promotion = 0;
//  // Horse moves
//  {
//    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
//             board[BB::Type::PROMOTED];
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      move.from = iterator.GetCurrentSquare();
//      moves = (getDiagRightAttacks(static_cast<Square>(move.from), occupied) |
//               getDiagLeftAttacks(static_cast<Square>(move.from), occupied) |
//               moveN(pieces) | moveE(pieces) | moveS(pieces) | moveW(pieces)) &
//              validMoves;
//      ourAttacks |= moves;
//      movesIterator.Init(moves);
//      while (movesIterator.Next()) {
//        move.to = movesIterator.GetCurrentSquare();
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//  }
//  // Dragon moves
//  {
//    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
//             board[BB::Type::PROMOTED];
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      move.from = iterator.GetCurrentSquare();
//      moves =
//          (getRankAttacks(static_cast<Square>(move.from), occupied) |
//           getFileAttacks(static_cast<Square>(move.from), occupied) |
//           moveNW(pieces) | moveNE(pieces) | moveSE(pieces) | moveSW(pieces)) &
//          validMoves;
//      ourAttacks |= moves;
//      movesIterator.Init(moves);
//      while (movesIterator.Next()) {
//        move.to = movesIterator.GetCurrentSquare();
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//  }
//  // King moves
//  {
//    pieces = board[BB::Type::KING] & board[BB::Type::ALL_BLACK];
//    iterator.Init(pieces);
//    while (iterator.Next()) {
//      move.from = iterator.GetCurrentSquare();
//      moves =
//          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
//           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
//          ~attackedByEnemy & ~board[BB::Type::ALL_BLACK];
//      ourAttacks |= moves;
//      movesIterator.Init(moves);
//      while (movesIterator.Next()) {
//        move.to = movesIterator.GetCurrentSquare();
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//  }
//  // Drop moves
//  {
//    Bitboard legalDropSpots;
//    // Pawns
//    if (board.inHand.pieceNumber.BlackPawn > 0) {
//      legalDropSpots = ~occupied;
//      // Cannot drop on last rank
//      legalDropSpots[TOP] &= ~TOP_RANK;
//      // Cannot drop to give checkmate
//      pieces = board[BB::Type::KING] & board[BB::Type::ALL_WHITE];
//      // All valid enemy king moves
//      moves =
//          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
//           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
//          ourAttacks;
//      // If there is only one spot pawn cannot block it
//      if (std::popcount<uint32_t>(moves[TOP]) +
//              std::popcount<uint32_t>(moves[MID]) +
//              std::popcount<uint32_t>(moves[BOTTOM]) ==
//          1) {
//        legalDropSpots &= ~moveS(moves);
//      }
//      // Cannot drop on file with other pawn
//      Bitboard validFiles;
//      for (int fileIdx = 0; fileIdx < BOARD_DIM; fileIdx++) {
//        Bitboard file = getFullFile(fileIdx);
//        if (!(file & board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
//              notPromoted)) {
//          validFiles |= file;
//        }
//      }
//      legalDropSpots &= validFiles;
//      move.from = BLACK_PAWN_DROP;
//      movesIterator.Init(legalDropSpots);
//      while (movesIterator.Next()) {
//        move.to = movesIterator.GetCurrentSquare();
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//    if (board.inHand.pieceNumber.BlackLance > 0) {
//      legalDropSpots = ~occupied;
//      // Cannot drop on last rank
//      legalDropSpots[TOP] &= ~TOP_RANK;
//      move.from = BLACK_LANCE_DROP;
//      movesIterator.Init(legalDropSpots);
//      while (movesIterator.Next()) {
//        move.to = movesIterator.GetCurrentSquare();
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//    if (board.inHand.pieceNumber.BlackKnight > 0) {
//      legalDropSpots = ~occupied;
//      // Cannot drop on last two ranks
//      legalDropSpots[TOP] &= TOP_RANK;
//      move.from = BLACK_KNIGHT_DROP;
//      movesIterator.Init(legalDropSpots);
//      while (movesIterator.Next()) {
//        move.to = movesIterator.GetCurrentSquare();
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//    legalDropSpots = ~occupied;
//    movesIterator.Init(legalDropSpots);
//    while (movesIterator.Next()) {
//      if (board.inHand.pieceNumber.BlackSilverGeneral > 0) {
//        move.from = BLACK_SILVER_GENERAL_DROP;
//        move.to = movesIterator.GetCurrentSquare();
//        *currentMove = move;
//        currentMove++;
//      }
//      if (board.inHand.pieceNumber.BlackGoldGeneral > 0) {
//        move.from = BLACK_GOLD_GENERAL_DROP;
//        move.to = movesIterator.GetCurrentSquare();
//        *currentMove = move;
//        currentMove++;
//      }
//      if (board.inHand.pieceNumber.BlackBishop > 0) {
//        move.from = BLACK_BISHOP_DROP;
//        move.to = movesIterator.GetCurrentSquare();
//        *currentMove = move;
//        currentMove++;
//      }
//      if (board.inHand.pieceNumber.BlackRook > 0) {
//        move.from = BLACK_ROOK_DROP;
//        move.to = movesIterator.GetCurrentSquare();
//        *currentMove = move;
//        currentMove++;
//      }
//    }
//  }
//}
//
//void makeMove(Board& board, const Move& move) {
//  Region toRegionIdx = squareToRegion(static_cast<Square>(move.to));
//  uint32_t toRegion = 1 << (REGION_SIZE - 1 - move.to % REGION_SIZE);
//  if (move.from < SQUARE_SIZE) {
//    Region fromRegionIdx = squareToRegion(static_cast<Square>(move.from));
//    uint32_t fromRegion = 1 << (REGION_SIZE - 1 - move.from % REGION_SIZE);
//    for (int i = 0; i < BB::Type::SIZE; i++) {
//      if (board[static_cast<BB::Type>(i)][toRegionIdx] & toRegion) {
//        board[static_cast<BB::Type>(i)][toRegionIdx] &= ~toRegion;
//      } else if (board[static_cast<BB::Type>(i)][fromRegionIdx] & fromRegion) {
//        board[static_cast<BB::Type>(i)][fromRegionIdx] &= ~fromRegion;
//        board[static_cast<BB::Type>(i)][toRegionIdx] |= toRegion;
//      }
//    }
//    if (move.promotion) {
//      board[BB::Type::PROMOTED][toRegionIdx] |= toRegion;
//    }
//  } else {
//    int offset = move.from - WHITE_PAWN_DROP;
//    uint64_t addedValue = 1 << offset;
//    board.inHand.value -= addedValue;
//    board[static_cast<BB::Type>(offset / 2)][toRegionIdx] |= toRegion;
//    board[static_cast<BB::Type>(BB::Type::ALL_WHITE + offset / 7)]
//         [toRegionIdx] |= toRegion;
//  }
//}
//
//void generateNextBoards(const Board& board,
//                        Move* movesArray,
//                        size_t length,
//                        Board* newBoardsArray) {
//  for (int i = 0; i < length; i++) {
//    newBoardsArray[i] = board;
//    makeMove(newBoardsArray[i], movesArray[i]);
//  }
//}
//
//void evaluateBoard(const Board& board, int16_t* valuesArray, uint32_t offset) {
//  int16_t valueWhite = 0;
//  int16_t valueBlack = 0;
//  // White
//  // non promoted pieces
//  valueWhite += popcount(board[BB::Type::ALL_WHITE] & board[BB::Type::PAWN] &
//                ~board[BB::Type::PROMOTED]) * PieceValue::PAWN;
//  valueWhite += popcount(board[BB::Type::ALL_WHITE] & board[BB::Type::KNIGHT] &
//                         ~board[BB::Type::PROMOTED]) *
//                PieceValue::KNIGHT;
//  valueWhite += popcount(board[BB::Type::ALL_WHITE] & board[BB::Type::LANCE] &
//                         ~board[BB::Type::PROMOTED]) *
//                PieceValue::LANCE;
//  valueWhite += popcount(board[BB::Type::ALL_WHITE] & board[BB::Type::SILVER_GENERAL] &
//               ~board[BB::Type::PROMOTED]) *
//      PieceValue::SILVER_GENERAL;
//  valueWhite += popcount(board[BB::Type::ALL_WHITE] & board[BB::Type::GOLD_GENERAL] &
//               ~board[BB::Type::PROMOTED]) *
//      PieceValue::GOLD_GENERAL;
//  valueWhite += popcount(board[BB::Type::ALL_WHITE] & board[BB::Type::BISHOP] &
//                         ~board[BB::Type::PROMOTED]) *
//                PieceValue::BISHOP;
//  valueWhite += popcount(board[BB::Type::ALL_WHITE] & board[BB::Type::ROOK] &
//                         ~board[BB::Type::PROMOTED]) *
//                PieceValue::ROOK;
//  // promoted pieces
//  valueWhite += popcount(board[BB::Type::ALL_WHITE] & board[BB::Type::PAWN] &
//                         board[BB::Type::PROMOTED]) *
//                PieceValue::PROMOTED_PAWN;
//  valueWhite += popcount(board[BB::Type::ALL_WHITE] & board[BB::Type::KNIGHT] &
//                         board[BB::Type::PROMOTED]) *
//                PieceValue::PROMOTED_KNIGHT;
//  valueWhite += popcount(board[BB::Type::ALL_WHITE] & board[BB::Type::LANCE] &
//                         board[BB::Type::PROMOTED]) *
//                PieceValue::PROMOTED_LANCE;
//  valueWhite +=
//      popcount(board[BB::Type::ALL_WHITE] & board[BB::Type::SILVER_GENERAL] &
//               board[BB::Type::PROMOTED]) *
//      PieceValue::PROMOTED_SILVER_GENERAL;
//  valueWhite += popcount(board[BB::Type::ALL_WHITE] & board[BB::Type::BISHOP] &
//                         board[BB::Type::PROMOTED]) *
//                PieceValue::PROMOTED_BISHOP;
//  valueWhite += popcount(board[BB::Type::ALL_WHITE] & board[BB::Type::ROOK] &
//                        ~board[BB::Type::PROMOTED]) *
//                PieceValue::PROMOTED_ROOK;
//  // in hand
//  valueWhite += board.inHand.pieceNumber.WhitePawn * IN_HAND_PAWN;
//  valueWhite += board.inHand.pieceNumber.WhiteLance * IN_HAND_LANCE;
//  valueWhite += board.inHand.pieceNumber.WhiteKnight * IN_HAND_KNIGHT;
//  valueWhite += board.inHand.pieceNumber.WhiteSilverGeneral * IN_HAND_SILVER_GENERAL;
//  valueWhite += board.inHand.pieceNumber.WhiteGoldGeneral * IN_HAND_GOLD_GENERAL;
//  valueWhite += board.inHand.pieceNumber.WhiteBishop * IN_HAND_BISHOP;
//  valueWhite += board.inHand.pieceNumber.WhiteRook * IN_HAND_ROOK;
//
//  // Black
//  // non promoted pieces
//  valueBlack += popcount(board[BB::Type::ALL_BLACK] & board[BB::Type::PAWN] &
//                         ~board[BB::Type::PROMOTED]) *
//                PieceValue::PAWN;
//  valueBlack += popcount(board[BB::Type::ALL_BLACK] & board[BB::Type::KNIGHT] &
//                         ~board[BB::Type::PROMOTED]) *
//                PieceValue::KNIGHT;
//  valueBlack += popcount(board[BB::Type::ALL_BLACK] & board[BB::Type::LANCE] &
//                         ~board[BB::Type::PROMOTED]) *
//                PieceValue::LANCE;
//  valueBlack +=
//      popcount(board[BB::Type::ALL_BLACK] & board[BB::Type::SILVER_GENERAL] &
//               ~board[BB::Type::PROMOTED]) *
//      PieceValue::SILVER_GENERAL;
//  valueBlack +=
//      popcount(board[BB::Type::ALL_BLACK] & board[BB::Type::GOLD_GENERAL] &
//               ~board[BB::Type::PROMOTED]) *
//      PieceValue::GOLD_GENERAL;
//  valueBlack += popcount(board[BB::Type::ALL_BLACK] & board[BB::Type::BISHOP] &
//                         ~board[BB::Type::PROMOTED]) *
//                PieceValue::BISHOP;
//  valueBlack += popcount(board[BB::Type::ALL_BLACK] & board[BB::Type::ROOK] &
//                         ~board[BB::Type::PROMOTED]) *
//                PieceValue::ROOK;
//  // promoted pieces
//  valueBlack += popcount(board[BB::Type::ALL_BLACK] & board[BB::Type::PAWN] &
//                         board[BB::Type::PROMOTED]) *
//                PieceValue::PROMOTED_PAWN;
//  valueBlack += popcount(board[BB::Type::ALL_BLACK] & board[BB::Type::KNIGHT] &
//                         board[BB::Type::PROMOTED]) *
//                PieceValue::PROMOTED_KNIGHT;
//  valueBlack += popcount(board[BB::Type::ALL_BLACK] & board[BB::Type::LANCE] &
//                         board[BB::Type::PROMOTED]) *
//                PieceValue::PROMOTED_LANCE;
//  valueBlack +=
//      popcount(board[BB::Type::ALL_BLACK] & board[BB::Type::SILVER_GENERAL] &
//               board[BB::Type::PROMOTED]) *
//      PieceValue::PROMOTED_SILVER_GENERAL;
//  valueBlack += popcount(board[BB::Type::ALL_BLACK] & board[BB::Type::BISHOP] &
//                         board[BB::Type::PROMOTED]) *
//                PieceValue::PROMOTED_BISHOP;
//  valueBlack += popcount(board[BB::Type::ALL_BLACK] & board[BB::Type::ROOK] &
//                         ~board[BB::Type::PROMOTED]) *
//                PieceValue::PROMOTED_ROOK;
//  // in hand
//  valueBlack += board.inHand.pieceNumber.BlackPawn * IN_HAND_PAWN;
//  valueBlack += board.inHand.pieceNumber.BlackLance * IN_HAND_LANCE;
//  valueBlack += board.inHand.pieceNumber.BlackKnight * IN_HAND_KNIGHT;
//  valueBlack +=
//      board.inHand.pieceNumber.BlackSilverGeneral * IN_HAND_SILVER_GENERAL;
//  valueBlack +=
//      board.inHand.pieceNumber.BlackGoldGeneral * IN_HAND_GOLD_GENERAL;
//  valueBlack += board.inHand.pieceNumber.BlackBishop * IN_HAND_BISHOP;
//  valueBlack += board.inHand.pieceNumber.BlackRook * IN_HAND_ROOK;
//
//  valuesArray[offset] =  valueWhite - valueBlack;
//}
//
//std::vector<Move> getAllLegalMoves(const Board& board, bool isWhite) {
//  Bitboard validMoves, attackedByEnemy;
//  std::vector<Move> moves;
//  if (isWhite) {
//    size_t movesCount = countWhiteMoves(board, validMoves, attackedByEnemy);
//    moves.resize(movesCount);
//    generateWhiteMoves(board, validMoves, attackedByEnemy, moves.data(), 0);
//  } else {
//    size_t movesCount = countBlackMoves(board, validMoves, attackedByEnemy);
//    moves.resize(movesCount);
//    generateBlackMoves(board, validMoves, attackedByEnemy, moves.data(), 0);
//  }
//  return moves;
//}
//
//std::vector<std::string> getAllLegalMovesUSI(const Board& board, bool isWhite) {
//  std::vector<Move> moves = getAllLegalMoves(board, isWhite);
//  std::vector<std::string> movesString(moves.size());
//  for (int i = 0; i < moves.size(); i++) {
//    movesString[i] = moveToString(moves[i]);
//  }
//  return movesString;
//}
//
//Move getBestMove(const Board& board,
//                 bool isWhite,
//                 unsigned int maxDepth,
//                 unsigned int maxTime) {
//  Bitboard validMoves, attackedByEnemy;
//  size_t movesCount;
//  std::vector<Move> moves;
//  if (isWhite) {
//    movesCount = countWhiteMoves(board, validMoves, attackedByEnemy);
//    moves.resize(movesCount);
//    generateWhiteMoves(board, validMoves, attackedByEnemy, moves.data(), 0);
//  } else {
//    movesCount = countBlackMoves(board, validMoves, attackedByEnemy);
//    moves.resize(movesCount);
//    generateBlackMoves(board, validMoves, attackedByEnemy, moves.data(), 0);
//  }
//
//  int randomMoveIdx = std::rand() % movesCount;
//  return moves[randomMoveIdx];
//}
//
//std::string getBestMoveUSI(const Board& board,
//                           bool isWhite,
//                           unsigned int maxDepth,
//                           unsigned int maxTime) {
//  return moveToString(getBestMove(board, isWhite, maxDepth, maxTime));
//}
//
//std::string moveToString(const Move& move) {
//  static const std::string pieceSymbols[14] = {
//      "p", "l", "n", "s", "g", "b", "r", "P", "L", "N", "S", "G", "B", "R"};
//  std::string moveString = "";
//  if (move.from >= WHITE_PAWN_DROP) {
//    moveString += pieceSymbols[move.from - WHITE_PAWN_DROP] + "*";
//  } else {
//    int fromFile = squareToFile(static_cast<Square>(move.from));
//    int fromRank = squareToRank(static_cast<Square>(move.from));
//    moveString += std::to_string(BOARD_DIM - fromFile) +
//                  static_cast<char>('a' + fromRank);
//  }
//  int toFile = squareToFile(static_cast<Square>(move.to));
//  int toRank = squareToRank(static_cast<Square>(move.to));
//  moveString +=
//      std::to_string(BOARD_DIM - toFile) + static_cast<char>('a' + toRank);
//  if (move.promotion) {
//    moveString += '+';
//  }
//  return moveString;
//}
//}  // namespace engine
//}  // namespace shogi