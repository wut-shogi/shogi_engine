#include "MoveGen.h"

size_t countAllMoves(const Board& board, bool isWhite) {
  Bitboard occupied = board[BB::Type::ALL_WHITE] | board[BB::Type::ALL_BLACK];
  Bitboard occupiedRot90 = board[BB::Type::OCCUPIED_ROT90];
  Bitboard occupiedRot45Right = board[BB::Type::OCCUPIED_ROTR45];
  Bitboard occupiedRot45Left = board[BB::Type::OCCUPIED_ROTL45];
  Bitboard promoted = board[BB::Type::PROMOTED];
  Bitboard notPromoted = ~promoted;

  int moveCount = 0;
  if (isWhite) {
    Bitboard playerMask = board[BB::Type::ALL_WHITE];
    Bitboard validMoves = ~playerMask;
    moveCount += countWhitePawnMoves(board[BB::Type::PAWN] & playerMask & notPromoted, validMoves);
    std::cout << "After pawns: " << moveCount << std::endl;
    moveCount += countWhiteKnightMoves(
        board[BB::Type::KNIGHT] & playerMask & notPromoted, validMoves);
    std::cout << "After knights: " << moveCount << std::endl;
    moveCount += countWhiteSilverGeneralMoves(
        board[BB::Type::SILVER_GENERAL] & playerMask & notPromoted, validMoves);
    std::cout << "After silverGen: " << moveCount << std::endl;
    moveCount +=
        countWhiteGoldGeneralMoves((board[BB::Type::GOLD_GENERAL] | ((board[BB::Type::PAWN] | board[BB::Type::KNIGHT] | board[BB::Type::SILVER_GENERAL] | board[BB::Type::LANCE]) & promoted)) & playerMask, validMoves);
    std::cout << "After goldGen: " << moveCount << std::endl;
    moveCount +=
        countKingMoves(board[BB::Type::KING] & playerMask, validMoves);
    std::cout << "After king: " << moveCount << std::endl;
    moveCount +=
        countWhiteLanceMoves(board[BB::Type::LANCE] & notPromoted & playerMask,
                             validMoves, occupiedRot90);
    std::cout << "After lance: " << moveCount << std::endl;
    moveCount +=
        countWhiteBishopMoves(board[BB::Type::BISHOP] & notPromoted & playerMask, validMoves, occupiedRot45Right, occupiedRot45Left);
    std::cout << "After bishop: " << moveCount << std::endl;
    moveCount +=
        countWhiteRookMoves(board[BB::Type::ROOK] & notPromoted & playerMask,
                            validMoves, occupied, occupiedRot90);
    std::cout << "After rook: " << moveCount << std::endl;
    moveCount +=
        countHorseMoves(board[BB::Type::BISHOP] & promoted & playerMask, validMoves,
                              occupiedRot45Right, occupiedRot45Left);
    std::cout << "After horse: " << moveCount << std::endl;
    moveCount += countDragonMoves(board[BB::Type::ROOK] & promoted & playerMask,
                                     validMoves, occupied, occupiedRot90);
    std::cout << "After dragon: " << moveCount << std::endl;
    moveCount += countDropMoves(board.inHandPieces.White, ~occupied,
                                board[BB::Type::PAWN] & playerMask,
                                board[BB::Type::PAWN] & (~playerMask), true);
    std::cout << "After drops: " << moveCount << std::endl;
  } else {
    Bitboard playerMask = board[BB::Type::ALL_BLACK];
    Bitboard validMoves = ~playerMask;
    moveCount += countBlackPawnMoves(
        board[BB::Type::PAWN] & playerMask & notPromoted, validMoves);
    std::cout << "After pawns: " << moveCount << std::endl;
    moveCount += countBlackKnightMoves(
        board[BB::Type::KNIGHT] & playerMask & notPromoted, validMoves);
    std::cout << "After knights: " << moveCount << std::endl;
    moveCount += countBlackSilverGeneralMoves(
        board[BB::Type::SILVER_GENERAL] & playerMask & notPromoted, validMoves);
    std::cout << "After silverGen: " << moveCount << std::endl;
    moveCount += countBlackGoldGeneralMoves(
        (board[BB::Type::GOLD_GENERAL] |
         ((board[BB::Type::PAWN] | board[BB::Type::KNIGHT] |
           board[BB::Type::SILVER_GENERAL] | board[BB::Type::LANCE]) &
          promoted)) & playerMask,
        validMoves);
    std::cout << "After goldGen: " << moveCount << std::endl;
    moveCount += countKingMoves(board[BB::Type::KING] & playerMask, validMoves);
    std::cout << "After king: " << moveCount << std::endl;
    moveCount +=
        countBlackLanceMoves(board[BB::Type::LANCE] & notPromoted & playerMask,
                             validMoves, occupiedRot90);
    std::cout << "After lance: " << moveCount << std::endl;
    moveCount += countBlackBishopMoves(
        board[BB::Type::BISHOP] & notPromoted & playerMask, validMoves,
        occupiedRot45Right, occupiedRot45Left);
    std::cout << "After bishop: " << moveCount << std::endl;
    moveCount +=
        countBlackRookMoves(board[BB::Type::ROOK] & notPromoted & playerMask,
                            validMoves, occupied, occupiedRot90);
    std::cout << "After rook: " << moveCount << std::endl;
    moveCount +=
        countHorseMoves(board[BB::Type::BISHOP] & promoted & playerMask,
                        validMoves, occupiedRot45Right, occupiedRot45Left);
    std::cout << "After horse: " << moveCount << std::endl;
    moveCount += countDragonMoves(board[BB::Type::ROOK] & promoted & playerMask,
                                  validMoves, occupied, occupiedRot90);
    std::cout << "After dragon: " << moveCount << std::endl;
    moveCount += countDropMoves(board.inHandPieces.Black, ~occupied,
                                board[BB::Type::PAWN] & playerMask,
                                board[BB::Type::PAWN] & (~playerMask), true);
    std::cout << "After drop: " << moveCount << std::endl;
  }

  return moveCount;
}