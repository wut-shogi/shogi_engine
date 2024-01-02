#pragma once
#include "Board.h"
#include "MoveGenHelpers.h"

namespace shogi {
namespace engine {
namespace GPU {
#ifdef __CUDACC__

int prefixSum(uint32_t* inValues, uint32_t inValuesLength);

int evaluateBoards(uint32_t size,
                   bool isWhite,
                   int16_t movesPerBoard,
                   Board* startBoard,
                   Move* inMoves,
                   int16_t* outValues);

int gatherValuesMax(uint32_t size,
                    uint16_t depth,
                    uint32_t* inOffsets,
                    int16_t* inValues,
                    int16_t* outValues,
                    uint32_t* bestIndex);

int gatherValuesMin(uint32_t size,
                    uint16_t depth,
                    uint32_t* inOffsets,
                    int16_t* inValues,
                    int16_t* outValues,
                    uint32_t* bestIndex);

int countWhiteMoves(uint32_t size,
                    int16_t movesPerBoard,
                    Board* startBoard,
                    Move* inMoves,
                    uint32_t inMovesSize,
                    uint32_t inMovesOffset,
                    uint32_t* outOffsets,
                    uint32_t* outBitboards);

int countBlackMoves(uint32_t size,
                    int16_t movesPerBoard,
                    Board* startBoard,
                    Move* inMoves,
                    uint32_t inMovesSize,
                    uint32_t inMovesOffset,
                    uint32_t* outOffsets,
                    uint32_t* outBitboards);

int generateWhiteMoves(uint32_t size,
                       int16_t movesPerBoard,
                       Board* startBoard,
                       Move* inMoves,
                       uint32_t inMovesSize,
                       uint32_t inMovesOffset,
                       uint32_t* inOffsets,
                       uint32_t* inBitboards,
                       Move* outMoves);

int generateBlackMoves(uint32_t size,
                       int16_t movesPerBoard,
                       Board* startBoard,
                       Move* inMoves,
                       uint32_t inMovesSize,
                       uint32_t inMovesOffset,
                       uint32_t* inOffsets,
                       uint32_t* inBitboards,
                       Move* outMoves);
#endif

}  // namespace GPU
}  // namespace engine
}  // namespace shogi