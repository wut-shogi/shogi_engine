#pragma once
#include <thrust/device_ptr.h>
#include "Board.h"

namespace shogi {
namespace engine {
namespace GPU {
int initLookUpArrays();
int countWhiteMoves(thrust::device_ptr<Board> inBoards,
                    uint32_t inBoardsLength,
                    thrust::device_ptr<Bitboard> outValidMoves,
                    thrust::device_ptr<Bitboard> outAttackedByEnemy,
                    thrust::device_ptr<Bitboard> outPinned,
                    thrust::device_ptr<uint32_t> outMovesOffset,
                    thrust::device_ptr<bool> isMate);

int countBlackMoves(thrust::device_ptr<Board> inBoards,
                    uint32_t inBoardsLength,
                    thrust::device_ptr<Bitboard> outValidMoves,
                    thrust::device_ptr<Bitboard> outAttackedByEnemy,
                    thrust::device_ptr<Bitboard> outPinned,
                    thrust::device_ptr<uint32_t> outMovesOffset,
                    thrust::device_ptr<bool> isMate);

int prefixSum(uint32_t* inValues, uint32_t inValuesLength);

int generateWhiteMoves(thrust::device_ptr<Board> inBoards,
                       uint32_t inBoardsLength,
                       thrust::device_ptr<Bitboard> inValidMoves,
                       thrust::device_ptr<Bitboard> inAttackedByEnemy,
                       thrust::device_ptr<Bitboard> inPinned,
                       thrust::device_ptr<uint32_t> inMovesOffset,
                       thrust::device_ptr<Move> outMoves,
                       thrust::device_ptr<uint32_t> outMoveToBoardIdx);

int generateBlackMoves(thrust::device_ptr<Board> inBoards,
                       uint32_t inBoardsLength,
                       thrust::device_ptr<Bitboard> inValidMoves,
                       thrust::device_ptr<Bitboard> inAttackedByEnemy,
                       thrust::device_ptr<Bitboard> inPinned,
                       thrust::device_ptr<uint32_t> inMovesOffset,
                       thrust::device_ptr<Move> outMoves,
                       thrust::device_ptr<uint32_t> outMoveToBoardIdx);

int generateWhiteBoards(thrust::device_ptr<Move> inMoves,
                        uint32_t inMovesLength,
                        thrust::device_ptr<Board> inBoards,
                        thrust::device_ptr<uint32_t> inMoveToBoardIdx,
                        thrust::device_ptr<Board> outBoards);

int generateBlackBoards(thrust::device_ptr<Move> inMoves,
                        uint32_t inMovesLength,
                        thrust::device_ptr<Board> inBoards,
                        thrust::device_ptr<uint32_t> inMoveToBoardIdx,
                        thrust::device_ptr<Board> outBoards);

int evaluateBoards(thrust::device_ptr<Board> inBoards,
                   uint32_t inBoardsLength,
                   thrust::device_ptr<int16_t> outValues);

int countWhiteMoves(uint32_t size,
                    int16_t movesPerBoard,
                    const Board& startBoard,
                    Move* inMoves,
                    uint32_t* outOffsets,
                    uint32_t* outBitboards);

int countBlackMoves(uint32_t size,
                    int16_t movesPerBoard,
                    const Board& startBoard,
                    Move* inMoves,
                    uint32_t* outOffsets,
                    uint32_t* outBitboards);

int generateWhiteMoves(uint32_t size,
                        int16_t movesPerBoard,
                        const Board& startBoard,
                        Move* inMoves,
                        uint32_t* inOffsets,
                        uint32_t* inBitboards,
                        Move* outMoves);

int generateBlackMoves(uint32_t size,
                        int16_t movesPerBoard,
                        const Board& startBoard,
                        Move* inMoves,
                        uint32_t* inOffsets,
                        uint32_t* inBitboards,
                        Move* outMoves);

int evaluateBoards(uint32_t size,
                   int16_t movesPerBoard,
                   const Board& startBoard,
                   Move* inMoves,
                   int16_t* outValues);
}  // namespace GPU
}  // namespace engine
}  // namespace shogi