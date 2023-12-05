#include "MoveGen.h"

//Bitboard findLegalMoves(const Board& board, bool isWhite) {
//  Bitboard enemyKing;
//  Bitboard nonSlidingCheckingPieces;
//  Bitboard pieces = board.bbs[BB::Type::PAWN] & board.bbs[BB::Type::ALL_WHITE];
//  nonSlidingCheckingPieces =
//      whitePawnsChecks(enemyKing, pieces);
//  pieces = board.bbs[BB::Type::PAWN] & board.bbs[BB::Type::ALL_WHITE];
//  
//}

size_t countAllMoves(const Board& board, bool isWhite) {
  Bitboard occupied = board[BB::Type::ALL_WHITE] | board[BB::Type::ALL_BLACK];
  Bitboard occupiedRot90 = board[BB::Type::OCCUPIED_ROT90];
  Bitboard occupiedRot45Right = board[BB::Type::OCCUPIED_ROTR45];
  Bitboard occupiedRot45Left = board[BB::Type::OCCUPIED_ROTL45];
  Bitboard promoted = board[BB::Type::PROMOTED];
  Bitboard notPromoted = ~promoted;
  BitboardIterator iterator;

  int moveCount = 0;
  if (isWhite) {
    Bitboard playerMask = board[BB::Type::ALL_WHITE];
    Bitboard validMoves = ~playerMask;
    moveCount += countWhitePawnsMoves(
        board[BB::Type::PAWN] & playerMask & notPromoted, validMoves);
    std::cout << "After pawns: " << moveCount << std::endl;
    moveCount += countWhiteKnightsMoves(
        board[BB::Type::KNIGHT] & playerMask & notPromoted, validMoves);
    std::cout << "After knights: " << moveCount << std::endl;
    moveCount += countWhiteSilverGeneralsMoves(
        board[BB::Type::SILVER_GENERAL] & playerMask & notPromoted, validMoves);
    std::cout << "After silverGen: " << moveCount << std::endl;
    moveCount += countWhiteGoldGeneralsMoves(
        (board[BB::Type::GOLD_GENERAL] |
         (board[BB::Type::PAWN] | board[BB::Type::KNIGHT] |
          board[BB::Type::SILVER_GENERAL] | board[BB::Type::LANCE]) &
             promoted) &
            playerMask,
        validMoves);
    std::cout << "After goldGen: " << moveCount << std::endl;
    moveCount += countKingMoves(board[BB::Type::KING] & playerMask, validMoves);
    std::cout << "After king: " << moveCount << std::endl;
    iterator = BitboardIterator(board[BB::Type::LANCE] & playerMask & notPromoted);
    while (iterator.Next()) {
      moveCount += countWhiteLancesMoves(iterator.GetCurrentSquare(), validMoves,
                                occupiedRot90);
    }
    std::cout << "After lance: " << moveCount << std::endl;
    iterator = BitboardIterator(board[BB::Type::BISHOP] & playerMask & notPromoted);
    while (iterator.Next()) {
      moveCount +=
          countWhiteBishopMoves(iterator.GetCurrentSquare(), validMoves,
                                occupiedRot45Right, occupiedRot45Left);
    }
    std::cout << "After bishop: " << moveCount << std::endl;
    iterator = BitboardIterator(board[BB::Type::ROOK] & playerMask & notPromoted);
    while (iterator.Next()) {
      moveCount += countWhiteRookMoves(iterator.GetCurrentSquare(), validMoves,
                                       occupied, occupiedRot90);
    }
    std::cout << "After rook: " << moveCount << std::endl;
    iterator =
        BitboardIterator(board[BB::Type::BISHOP] & playerMask & promoted);
    while (iterator.Next()) {
      moveCount += countHorseMoves(iterator.GetCurrentSquare(), validMoves,
                                   occupiedRot45Right, occupiedRot45Left);
    }
    std::cout << "After horse: " << moveCount << std::endl;
    iterator =
        BitboardIterator(board[BB::Type::ROOK] & playerMask & promoted);
    while (iterator.Next()) {
      moveCount += countDragonMoves(iterator.GetCurrentSquare(), validMoves,
                                    occupied, occupiedRot90);
    }
    std::cout << "After dragon: " << moveCount << std::endl;
    moveCount += countDropMoves(board.inHandPieces.White, ~occupied,
                                board[BB::Type::PAWN] & playerMask,
                                board[BB::Type::PAWN] & (~playerMask), true);
    std::cout << "After drops: " << moveCount << std::endl;
  } else {
    Bitboard playerMask = board[BB::Type::ALL_BLACK];
    Bitboard validMoves = ~playerMask;
    moveCount += countBlackPawnsMoves(
        board[BB::Type::PAWN] & playerMask & notPromoted, validMoves);
    std::cout << "After pawns: " << moveCount << std::endl;
    moveCount += countBlackKnightsMoves(
        board[BB::Type::KNIGHT] & playerMask & notPromoted, validMoves);
    std::cout << "After knights: " << moveCount << std::endl;
    moveCount += countBlackSilverGeneralsMoves(
        board[BB::Type::SILVER_GENERAL] & playerMask & notPromoted, validMoves);
    std::cout << "After silverGen: " << moveCount << std::endl;
    moveCount += countBlackGoldGeneralsMoves(
        (board[BB::Type::GOLD_GENERAL] |
         (board[BB::Type::PAWN] | board[BB::Type::KNIGHT] |
          board[BB::Type::SILVER_GENERAL] | board[BB::Type::LANCE]) &
             promoted) &
            playerMask,
        validMoves);
    std::cout << "After goldGen: " << moveCount << std::endl;
    moveCount += countKingMoves(board[BB::Type::KING] & playerMask, validMoves);
    std::cout << "After king: " << moveCount << std::endl;
    iterator =
        BitboardIterator(board[BB::Type::LANCE] & playerMask & notPromoted);
    while (iterator.Next()) {
      moveCount += countBlackLancesMoves(iterator.GetCurrentSquare(),
                                         validMoves, occupiedRot90);
    }
    std::cout << "After lance: " << moveCount << std::endl;
    iterator =
        BitboardIterator(board[BB::Type::BISHOP] & playerMask & notPromoted);
    while (iterator.Next()) {
      moveCount +=
          countBlackBishopMoves(iterator.GetCurrentSquare(), validMoves,
                                occupiedRot45Right, occupiedRot45Left);
    }
    std::cout << "After bishop: " << moveCount << std::endl;
    iterator =
        BitboardIterator(board[BB::Type::ROOK] & playerMask & notPromoted);
    while (iterator.Next()) {
      moveCount += countBlackRookMoves(iterator.GetCurrentSquare(), validMoves,
                                       occupied, occupiedRot90);
    }
    std::cout << "After rook: " << moveCount << std::endl;
    iterator =
        BitboardIterator(board[BB::Type::BISHOP] & playerMask & promoted);
    while (iterator.Next()) {
      moveCount += countHorseMoves(iterator.GetCurrentSquare(), validMoves,
                                   occupiedRot45Right, occupiedRot45Left);
    }
    std::cout << "After horse: " << moveCount << std::endl;
    iterator = BitboardIterator(board[BB::Type::ROOK] & playerMask & promoted);
    while (iterator.Next()) {
      moveCount += countDragonMoves(iterator.GetCurrentSquare(), validMoves,
                                    occupied, occupiedRot90);
    }
    std::cout << "After dragon: " << moveCount << std::endl;
    moveCount += countDropMoves(board.inHandPieces.Black, ~occupied,
                                board[BB::Type::PAWN] & playerMask,
                                board[BB::Type::PAWN] & (~playerMask), true);
    std::cout << "After drop: " << moveCount << std::endl;
  }

  return moveCount;
}


std::vector<std::pair<int, int>> getLegalMovesFromSquare(std::string SFENstring,
    int rank,
    int file) {
  std::vector<std::pair<int, int>> result;
  //Board board = Board::FromSFEN(SFENstring);
  return result;
}