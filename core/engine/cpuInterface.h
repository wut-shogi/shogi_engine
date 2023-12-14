#pragma once
#include "Board.h"

namespace shogi {
namespace engine {
namespace CPU {
void countWhiteMoves(Board* inBoards,
                     uint32_t inBoardsLength,
                     Bitboard* outValidMoves,
                     Bitboard* outAttackedByEnemy,
                     Bitboard* outPinned,
                     uint32_t* outMovesOffset,
                     bool* isMate);

void countBlackMoves(Board* inBoards,
                     uint32_t inBoardsLength,
                     Bitboard* outValidMoves,
                     Bitboard* outAttackedByEnemy,
                     Bitboard* outPinned,
                     uint32_t* outMovesOffset,
                     bool* isMate);

void prefixSum(uint32_t* inValues, uint32_t inValuesLength);

void generateWhiteMoves(Board* inBoards,
                        uint32_t inBoardsLength,
                        Bitboard* inValidMoves,
                        Bitboard* inAttackedByEnemy,
                        Bitboard* inPinned,
                        uint32_t* inMovesOffset,
                        Move* outMoves,
                        uint32_t* outMoveToBoardIdx);

void generateBlackMoves(Board* inBoards,
                        uint32_t inBoardsLength,
                        Bitboard* inValidMoves,
                        Bitboard* inAttackedByEnemy,
                        Bitboard* inPinned,
                        uint32_t* inMovesOffset,
                        Move* outMoves,
                        uint32_t* outMoveToBoardIdx);

void generateWhiteBoards(Move* inMoves,
                         uint32_t inMovesLength,
                         Board* inBoards,
                         uint32_t* moveToBoardIdx,
                         Board* outBoards);

void generateBlackBoards(Move* inMoves,
                         uint32_t inMovesLength,
                         Board* inBoards,
                         uint32_t* moveToBoardIdx,
                         Board* outBoards);

void evaluateBoards(Board* inBoards,
                    uint32_t inBoardsLength,
                    int16_t* outValues);

}  // namespace CPU
}  // namespace engine
}  // namespace shogi