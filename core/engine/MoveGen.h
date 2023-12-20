#pragma once
#include <string>
#include <vector>
#include "Board.h"
namespace shogi {
namespace engine {
__host__ __device__ void getWhitePiecesInfo(const Board& board,
                                            Bitboard& outPinned,
                                            Bitboard& outValidMoves,
                                            Bitboard& outAttackedByEnemy);

__host__ __device__ void getBlackPiecesInfo(const Board& board,
                                            Bitboard& outPinned,
                                            Bitboard& outValidMoves,
                                            Bitboard& outAttackedByEnemy);

__host__ __device__ uint32_t countWhiteMoves(const Board& board,
                                             Bitboard& pinned,
                                             Bitboard& validMoves,
                                             Bitboard& attackedByEnemy);

__host__ __device__ uint32_t countBlackMoves(const Board& board,
                                             Bitboard& pinned,
                                             Bitboard& validMoves,
                                             Bitboard& attackedByEnemy);

__host__ __device__ uint32_t generateWhiteMoves(const Board& board,
                                            Bitboard& pinned,
                                            Bitboard& validMoves,
                                            Bitboard& attackedByEnemy, Move* movesArray);

__host__ __device__ uint32_t generateBlackMoves(const Board& board,
                                            Bitboard& pinned,
                                            Bitboard& validMoves,
                                            Bitboard& attackedByEnemy, Move* movesArray);
}  // namespace engine
}  // namespace shogi
