#pragma once
#include <string>
#include <vector>
#include "Board.h"
namespace shogi {
namespace engine {
RUNTYPE void getWhitePiecesInfo(const Board& board,
                                            Bitboard& outPinned,
                                            Bitboard& outValidMoves,
                                            Bitboard& outAttackedByEnemy);

RUNTYPE void getBlackPiecesInfo(const Board& board,
                                            Bitboard& outPinned,
                                            Bitboard& outValidMoves,
                                            Bitboard& outAttackedByEnemy);

RUNTYPE uint32_t countWhiteMoves(const Board& board,
                                             Bitboard& pinned,
                                             Bitboard& validMoves,
                                             Bitboard& attackedByEnemy);

RUNTYPE uint32_t countBlackMoves(const Board& board,
                                             Bitboard& pinned,
                                             Bitboard& validMoves,
                                             Bitboard& attackedByEnemy);

RUNTYPE uint32_t generateWhiteMoves(const Board& board,
                                            Bitboard& pinned,
                                            Bitboard& validMoves,
                                            Bitboard& attackedByEnemy, Move* movesArray);

RUNTYPE uint32_t generateBlackMoves(const Board& board,
                                            Bitboard& pinned,
                                            Bitboard& validMoves,
                                            Bitboard& attackedByEnemy, Move* movesArray);
}  // namespace engine
}  // namespace shogi
