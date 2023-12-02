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
        (board[BB::Type::GOLD_GENERAL] | (board[BB::Type::PAWN] |
         board[BB::Type::KNIGHT] | board[BB::Type::SILVER_GENERAL] |
         Bitboard(static_cast<Square>(board.nonBitboardPieces.White.Lance1)) |
         Bitboard(static_cast<Square>(board.nonBitboardPieces.White.Lance2))) &
            promoted) & playerMask,
        validMoves);
    std::cout << "After goldGen: " << moveCount << std::endl;
    moveCount += countKingMoves(
        static_cast<Square>(board.nonBitboardPieces.White.King), validMoves);
    std::cout << "After king: " << moveCount << std::endl;
    moveCount += countWhiteLancesMoves(
        static_cast<Square>(board.nonBitboardPieces.White.Lance1),
        static_cast<Square>(board.nonBitboardPieces.White.Lance2), validMoves,
        occupiedRot90);
    std::cout << "After lance: " << moveCount << std::endl;
    moveCount += countWhiteBishopMoves(
        static_cast<Square>(board.nonBitboardPieces.White.Bishop), validMoves,
        occupiedRot45Right, occupiedRot45Left);
    std::cout << "After bishop: " << moveCount << std::endl;
    moveCount += countWhiteRookMoves(
        static_cast<Square>(board.nonBitboardPieces.White.Rook), validMoves,
        occupied, occupiedRot90);
    std::cout << "After rook: " << moveCount << std::endl;
    moveCount += countHorseMoves(
        static_cast<Square>(board.nonBitboardPieces.White.Horse), validMoves,
        occupiedRot45Right, occupiedRot45Left);
    std::cout << "After horse: " << moveCount << std::endl;
    moveCount += countDragonMoves(
        static_cast<Square>(board.nonBitboardPieces.White.Dragon), validMoves,
        occupied, occupiedRot90);
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
        (board[BB::Type::GOLD_GENERAL] | (board[BB::Type::PAWN] |
         board[BB::Type::KNIGHT] | board[BB::Type::SILVER_GENERAL] |
         Bitboard(static_cast<Square>(board.nonBitboardPieces.Black.Lance1)) |
         Bitboard(static_cast<Square>(board.nonBitboardPieces.Black.Lance2))) &
            promoted) & playerMask,
        validMoves);
    std::cout << "After goldGen: " << moveCount << std::endl;
    moveCount += countKingMoves(
        static_cast<Square>(board.nonBitboardPieces.Black.King), validMoves);
    std::cout << "After king: " << moveCount << std::endl;
    moveCount += countBlackLancesMoves(
        static_cast<Square>(board.nonBitboardPieces.Black.Lance1),
        static_cast<Square>(board.nonBitboardPieces.Black.Lance2), validMoves,
        occupiedRot90);
    std::cout << "After lance: " << moveCount << std::endl;
    moveCount += countBlackBishopMoves(
        static_cast<Square>(board.nonBitboardPieces.Black.Bishop), validMoves,
        occupiedRot45Right, occupiedRot45Left);
    std::cout << "After bishop: " << moveCount << std::endl;
    moveCount += countBlackRookMoves(
        static_cast<Square>(board.nonBitboardPieces.Black.Rook), validMoves,
        occupied, occupiedRot90);
    std::cout << "After rook: " << moveCount << std::endl;
    moveCount += countHorseMoves(
        static_cast<Square>(board.nonBitboardPieces.Black.Horse), validMoves,
        occupiedRot45Right, occupiedRot45Left);
    std::cout << "After horse: " << moveCount << std::endl;
    moveCount += countDragonMoves(
        static_cast<Square>(board.nonBitboardPieces.Black.Dragon), validMoves,
        occupied, occupiedRot90);
    std::cout << "After dragon: " << moveCount << std::endl;
    moveCount += countDropMoves(board.inHandPieces.Black, ~occupied,
                                board[BB::Type::PAWN] & playerMask,
                                board[BB::Type::PAWN] & (~playerMask), true);
    std::cout << "After drop: " << moveCount << std::endl;
  }

  return moveCount;
}