#pragma once
#include <thrust/device_ptr.h>
#include "Board.h"

namespace shogi {
namespace engine {
namespace GPU {
int prefixSum(uint32_t* inValues, uint32_t inValuesLength);

int countWhiteMoves(uint32_t size,
                    int16_t movesPerBoard,
                    Board* startBoard,
                    Move* inMoves,
                    uint32_t* outOffsets,
                    uint32_t* outBitboards);

int countBlackMoves(uint32_t size,
                    int16_t movesPerBoard,
                    Board* startBoard,
                    Move* inMoves,
                    uint32_t* outOffsets,
                    uint32_t* outBitboards);

int generateWhiteMoves(uint32_t size,
                       int16_t movesPerBoard,
                       Board* startBoard,
                       Move* inMoves,
                       uint32_t* inOffsets,
                       uint32_t* inBitboards,
                       Move* outMoves);

int generateBlackMoves(uint32_t size,
                       int16_t movesPerBoard,
                       Board* startBoard,
                       Move* inMoves,
                       uint32_t* inOffsets,
                       uint32_t* inBitboards,
                       Move* outMoves);

int evaluateBoards(uint32_t size,
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
}  // namespace GPU
}  // namespace engine
}  // namespace shogi