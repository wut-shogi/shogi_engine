#pragma once
#include <string>
#include <vector>
#include "Board.h"
namespace shogi {
namespace engine {
__host__ __device__ void getWhitePiecesInfo(Board& board,
                                            Bitboard& outPinned,
                                            Bitboard& outValidMoves,
                                            Bitboard& outAttackedByEnemy);

__host__ __device__ void getBlackPiecesInfo(Board& board,
                                            Bitboard& outPinned,
                                            Bitboard& outValidMoves,
                                            Bitboard& outAttackedByEnemy);

__host__ __device__ uint32_t countWhiteMoves(Board& board,
                                             Bitboard& pinned,
                                             Bitboard& validMoves,
                                             Bitboard& attackedByEnemy);

__host__ __device__ uint32_t countBlackMoves(Board& board,
                                             Bitboard& pinned,
                                             Bitboard& validMoves,
                                             Bitboard& attackedByEnemy);

__host__ __device__ uint32_t generateWhiteMoves(Board& board,
                                            Bitboard& pinned,
                                            Bitboard& validMoves,
                                            Bitboard& attackedByEnemy, Move* movesArray);

__host__ __device__ uint32_t generateBlackMoves(Board& board,
                                            Bitboard& pinned,
                                            Bitboard& validMoves,
                                            Bitboard& attackedByEnemy, Move* movesArray);
}  // namespace engine
}  // namespace shogi
