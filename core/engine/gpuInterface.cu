#include <device_launch_parameters.h>
#include <stdio.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include "MoveGenHelpers.h"
#include "gpuInterface.h"

namespace shogi {
namespace engine {

#define THREADS_COUNT 32

namespace GPU {
#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code,
                      const char* file,
                      int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}
#define ARRAY_SIZE 10368
struct LookUpTables {
  Bitboard* rankAttacks;
  Bitboard* fileAttacks;
  Bitboard* diagRightAttacks;
  Bitboard* diagLeftAttacks;

  Bitboard* rankMask;
  Bitboard* fileMask;

  uint32_t* startSqDiagRight;
  uint32_t* startSqDiagLeft;
};

__device__ __constant__ LookUpTables lookUpTables[1];

int initLookUpArrays() {
  LookUpTables lookUpsTables_Host;
  cudaMalloc((void**)&lookUpsTables_Host.rankAttacks,
             ARRAY_SIZE * sizeof(Bitboard));
  cudaMalloc((void**)&lookUpsTables_Host.fileAttacks,
             ARRAY_SIZE * sizeof(Bitboard));
  cudaMalloc((void**)&lookUpsTables_Host.diagRightAttacks,
             ARRAY_SIZE * sizeof(Bitboard));
  cudaMalloc((void**)&lookUpsTables_Host.diagLeftAttacks,
             ARRAY_SIZE * sizeof(Bitboard));
  cudaMalloc((void**)&lookUpsTables_Host.rankMask, 9 * sizeof(Bitboard));
  cudaMalloc((void**)&lookUpsTables_Host.fileMask, 9 * sizeof(Bitboard));
  cudaMalloc((void**)&lookUpsTables_Host.startSqDiagRight,
             81 * sizeof(uint32_t));
  cudaMalloc((void**)&lookUpsTables_Host.startSqDiagLeft,
             81 * sizeof(uint32_t));

  cudaMemcpy(lookUpsTables_Host.rankAttacks, CPU::getRankAttacksPtr(),
             ARRAY_SIZE * sizeof(Bitboard), cudaMemcpyHostToDevice);
  cudaMemcpy(lookUpsTables_Host.fileAttacks, CPU::getFileAttacksPtr(),
             ARRAY_SIZE * sizeof(Bitboard), cudaMemcpyHostToDevice);
  cudaMemcpy(lookUpsTables_Host.diagRightAttacks, CPU::getDiagRightAttacksPtr(),
             ARRAY_SIZE * sizeof(Bitboard), cudaMemcpyHostToDevice);
  cudaMemcpy(lookUpsTables_Host.diagLeftAttacks, CPU::getDiagLeftAttacksPtr(),
             ARRAY_SIZE * sizeof(Bitboard), cudaMemcpyHostToDevice);
  cudaMemcpy(lookUpsTables_Host.rankMask, CPU::getRankMaskPtr(),
             9 * sizeof(Bitboard), cudaMemcpyHostToDevice);
  cudaMemcpy(lookUpsTables_Host.fileMask, CPU::getFileMaskPtr(),
             9 * sizeof(Bitboard), cudaMemcpyHostToDevice);

  const uint32_t startingSquareDiagRightTemplate[BOARD_SIZE] = {
      0, 1,  2,  3,  4,  5,  6,  7,  8,   //
      1, 2,  3,  4,  5,  6,  7,  8,  17,  //
      2, 3,  4,  5,  6,  7,  8,  17, 26,  //
      3, 4,  5,  6,  7,  8,  17, 26, 35,  //
      4, 5,  6,  7,  8,  17, 26, 35, 44,  //
      5, 6,  7,  8,  17, 26, 35, 44, 53,  //
      6, 7,  8,  17, 26, 35, 44, 53, 62,  //
      7, 8,  17, 26, 35, 44, 53, 62, 71,  //
      8, 17, 26, 35, 44, 53, 62, 71, 80,
  };
  cudaMemcpy(lookUpsTables_Host.startSqDiagRight,
             startingSquareDiagRightTemplate, 81 * sizeof(uint32_t),
             cudaMemcpyHostToDevice);

  const uint32_t startingSquareDiagLeftTemplate[BOARD_SIZE] = {
      0,  1,  2,  3,  4,  5,  6,  7, 8,  //
      9,  0,  1,  2,  3,  4,  5,  6, 7,  //
      18, 9,  0,  1,  2,  3,  4,  5, 6,  //
      27, 18, 9,  0,  1,  2,  3,  4, 5,  //
      36, 27, 18, 9,  0,  1,  2,  3, 4,  //
      45, 36, 27, 18, 9,  0,  1,  2, 3,  //
      54, 45, 36, 27, 18, 9,  0,  1, 2,  //
      63, 54, 45, 36, 27, 18, 9,  0, 1,  //
      72, 63, 54, 45, 36, 27, 18, 9, 0,
  };
  cudaMemcpy(lookUpsTables_Host.startSqDiagLeft, startingSquareDiagLeftTemplate,
             81 * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaError_t cudaStatus;
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cuda alloc and memcpy launch failed: %s\n",
            cudaGetErrorString(cudaStatus));
    return -1;
  }
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr,
            "cudaDeviceSynchronize returned error code %d after launching "
            "cuda alloc and memcpy!\n",
            cudaStatus);
    return -1;
  }

  cudaMemcpyToSymbol(GPU::lookUpTables, &lookUpsTables_Host,
                     sizeof(LookUpTables), 0, cudaMemcpyHostToDevice);

  cudaStatus;
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cuda alloc and memcpy launch failed: %s\n",
            cudaGetErrorString(cudaStatus));
    return -1;
  }
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr,
            "cudaDeviceSynchronize returned error code %d after launching "
            "cuda alloc and memcpy!\n",
            cudaStatus);
    return -1;
  }
  return 0;
}

__device__ uint32_t getDiagRightBlockPattern(const Bitboard& occupied,
                                             Square square) {
  uint32_t result = 0;
  uint32_t startingSquare = lookUpTables[0].startSqDiagRight[square];
  int len = startingSquare > 9 ? 7 - startingSquare / 9 : startingSquare - 1;
  for (int i = 0; i < len; i++) {
    result += occupied.GetBit(static_cast<Square>(startingSquare + i * SW + SW))
              << i;
  }
  return result;
}

__device__ uint32_t getDiagLeftBlockPattern(const Bitboard& occupied,
                                            Square square) {
  uint32_t result = 0;
  uint32_t startingSquare = lookUpTables[0].startSqDiagLeft[square];
  int len =
      startingSquare > 9 ? 7 - startingSquare / 9 : 7 - startingSquare % 9;
  for (int i = 0; i < len; i++) {
    result += occupied.GetBit(static_cast<Square>(startingSquare + i * SE + SE))
              << i;
  }
  return result;
}

__device__ const Bitboard& getRankAttacks(const Square& square,
                                          const Bitboard& occupied) {
  return lookUpTables[0]
      .rankAttacks[square * 128 + getRankBlockPattern(occupied, square)];
}

__device__ const Bitboard& getFileAttacks(const Square& square,
                                          const Bitboard& occupied) {
  return lookUpTables[0]
      .fileAttacks[square * 128 + getFileBlockPattern(occupied, square)];
}
__device__ const Bitboard& getDiagRightAttacks(const Square& square,
                                               const Bitboard& occupied) {
  return lookUpTables[0]
      .diagRightAttacks[square * 128 +
                        getDiagRightBlockPattern(occupied, square)];
}
__device__ const Bitboard& getDiagLeftAttacks(const Square& square,
                                              const Bitboard& occupied) {
  return lookUpTables[0].diagLeftAttacks[square * 128 + getDiagLeftBlockPattern(
                                                            occupied, square)];
}
__device__ const Bitboard& getRankMask(const uint32_t& rank) {
  return lookUpTables[0].rankMask[rank];
}
__device__ const Bitboard& getFileMask(const uint32_t& file) {
  return lookUpTables[0].fileMask[file];
}
}  // namespace GPU

int calculateNumberOfBlocks(uint32_t size) {
  int numberOfBlocks = (int)ceil(size / (double)THREADS_COUNT);
  std::cout << "Number of blocks: " << numberOfBlocks << std::endl;
  return numberOfBlocks;
}

__global__ void countWhiteMovesKernel(Board* inBoards,
                                      uint32_t inBoardsLength,
                                      Bitboard* outValidMoves,
                                      Bitboard* outAttackedByEnemy,
                                      Bitboard* outPinned,
                                      uint32_t* outMovesOffset,
                                      bool* isMate) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= inBoardsLength)
    return;

  Board board = inBoards[index];
  Bitboard king = board[BB::Type::KING] & board[BB::Type::ALL_WHITE];
  Bitboard notPromoted = ~board[BB::Type::PROMOTED];
  Bitboard checkingPieces, attacked, pieces, moves, slidingChecksPaths, attacks,
      attacksFull, mask, potentialPin, pinned, ourAttacks;
  BitboardIterator iterator;
  Square square;
  size_t numberOfMoves = 0;

  // Non Sliding pieces
  // Pawns
  pieces =
      board.bbs[BB::Type::PAWN] & board.bbs[BB::Type::ALL_BLACK] & notPromoted;
  checkingPieces |= moveS(king) & pieces;
  attacked |= moveN(pieces);
  // Knights
  pieces = board.bbs[BB::Type::KNIGHT] & board.bbs[BB::Type::ALL_BLACK] &
           notPromoted;
  checkingPieces |= moveS(moveSE(king) | moveSW(king)) & pieces;
  attacked |= moveN(moveNE(pieces) | moveNW(pieces));
  // Silve generals
  pieces = board.bbs[BB::Type::SILVER_GENERAL] &
           board.bbs[BB::Type::ALL_BLACK] & notPromoted;
  checkingPieces |= (moveSE(king) | moveS(king) | moveSW(king) | moveNE(king) |
                     moveNW(king)) &
                    pieces;
  attacked |= moveNE(pieces) | moveN(pieces) | moveNW(pieces) | moveSE(pieces) |
              moveSW(pieces);
  // Gold generals
  pieces = (board[BB::Type::GOLD_GENERAL] |
            ((board[BB::Type::PAWN] | board[BB::Type::KNIGHT] |
              board[BB::Type::SILVER_GENERAL] | board[BB::Type::LANCE]) &
             board[BB::Type::PROMOTED])) &
           board.bbs[BB::Type::ALL_BLACK];
  checkingPieces |= (moveSE(king) | moveS(king) | moveSW(king) | moveE(king) |
                     moveW(king) | moveN(king)) &
                    pieces;
  attacked |= moveNE(pieces) | moveN(pieces) | moveNW(pieces) | moveE(pieces) |
              moveW(pieces) | moveS(pieces);
  // Horse (non sliding part)
  pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
           board[BB::Type::PROMOTED];
  checkingPieces |=
      (moveN(king) | moveE(king) | moveS(king) | moveW(king)) & pieces;
  attacked |= moveN(pieces) | moveE(pieces) | moveS(pieces) | moveW(pieces);
  // Dragon (non sliding part)
  pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
           board[BB::Type::PROMOTED];
  checkingPieces |=
      (moveNW(king) | moveNE(king) | moveSE(king) | moveSW(king)) & pieces;
  attacked |= moveNW(pieces) | moveNE(pieces) | moveSE(pieces) | moveSW(pieces);

  // Sliding pieces
  iterator.Init(king);
  iterator.d_Next();
  Square kingSquare = iterator.GetCurrentSquare();
  Bitboard occupied = board[BB::Type::ALL_BLACK] | board[BB::Type::ALL_WHITE];
  // Lance
  {
    pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] & notPromoted;
    checkingPieces |= GPU::getFileAttacks(kingSquare, occupied) &
                      ~GPU::getRankMask(squareToRank(kingSquare)) & pieces;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      attacksFull = GPU::getFileAttacks(square, occupied);
      attacked |= attacksFull;
      mask = GPU::getRankMask(squareToRank(square));
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::ALL_WHITE];
        attacks = GPU::getFileAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
    }
  }

  // Rook and dragon
  {
    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK];
    checkingPieces |= (GPU::getRankAttacks(kingSquare, occupied) |
                       GPU::getFileAttacks(kingSquare, occupied)) &
                      pieces;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      // Check if king is in check without white pieces
      // We have to check all 4 directions
      // left-right
      attacksFull = GPU::getRankAttacks(square, occupied);
      attacked |= attacksFull;
      mask = GPU::getFileMask(squareToFile(square));
      // left
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & mask;
        attacks = GPU::getRankAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
      // right
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & ~mask;
        attacks = GPU::getRankAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & ~mask;
      }
      // up-down
      attacksFull = GPU::getFileAttacks(square, occupied);
      attacked |= attacksFull;
      mask = GPU::getRankMask(squareToRank(square));
      // up
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & mask;
        attacks = GPU::getFileAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
      // down
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & ~mask;
        attacks = GPU::getFileAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & ~mask;
      }
    }
  }

  // Bishop and horse pins
  {
    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK];
    checkingPieces |= (GPU::getDiagRightAttacks(kingSquare, occupied) |
                       GPU::getDiagLeftAttacks(kingSquare, occupied)) &
                      pieces;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      // Check if king is in check without white pieces
      // We have to check all 4 directions
      // right diag
      attacksFull = GPU::getDiagRightAttacks(square, occupied);
      attacked |= attacksFull;
      mask = ~GPU::getFileMask(squareToFile(square)) &
             GPU::getRankMask(squareToRank(square));
      // SW
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & mask;
        attacks = GPU::getDiagRightAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
      // NE
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & ~mask;
        attacks = GPU::getDiagRightAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & ~mask;
      }
      // left diag
      attacksFull = GPU::getDiagLeftAttacks(square, occupied);
      attacked |= attacksFull;
      mask = GPU::getFileMask(squareToFile(square)) &
             GPU::getRankMask(squareToRank(square));
      // NW
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & mask;
        attacks = GPU::getDiagLeftAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
      // SE
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & ~mask;
        attacks = GPU::getDiagLeftAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & ~mask;
      }
    }
  }

  int numberOfCheckingPieces = d_popcount(checkingPieces[TOP]) +
                               d_popcount(checkingPieces[MID]) +
                               d_popcount(checkingPieces[BOTTOM]);

  // King can always move to non attacked squares
  moves = moveNE(king) | moveN(king) | moveNW(king) | moveE(king) |
          moveW(king) | moveSE(king) | moveS(king) | moveSW(king);
  moves &= ~attacked & ~board[BB::Type::ALL_WHITE];
  numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                   d_popcount(moves[BOTTOM]);

  Bitboard validMoves;
  // If more then one piece is checking the king and king cannot move its mate
  if (numberOfCheckingPieces > 1) {
    if (numberOfMoves == 0) {
      *isMate = true;
      return;
    }
  } else if (numberOfCheckingPieces == 1) {
    // if king is checked by exactly one piece legal moves can also be block
    // sliding check or capture a checking piece
    validMoves = checkingPieces | (slidingChecksPaths & ~king);
  } else if (numberOfCheckingPieces == 0) {
    // If there is no checks all moves are valid (you cannot capture your own
    // piece)
    validMoves = ~board[BB::Type::ALL_WHITE];
  }

  outValidMoves[index] = validMoves;
  outAttackedByEnemy[index] = attacked;
  outPinned[index] = pinned;

  // Pawn moves
  {
    pieces = ~pinned & board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
             notPromoted;
    moves = moveS(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves +=
        d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
        d_popcount(moves[BOTTOM] & ~BOTTOM_RANK) * 2 +  // promotions
        d_popcount(moves[BOTTOM] & BOTTOM_RANK);        // forced promotion
  }

  // Knight moves
  {
    pieces = ~pinned & board[BB::Type::KNIGHT] & board[BB::Type::ALL_WHITE] &
             notPromoted;
    moves = moveS(moveSE(pieces)) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM] & TOP_RANK) * 2 +  // promotions
                     d_popcount(moves[BOTTOM] & ~TOP_RANK);  // forced promotion
    moves = moveS(moveSW(pieces)) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM] & TOP_RANK) * 2 +  // promotions
                     d_popcount(moves[BOTTOM] & ~TOP_RANK);  // forced promotion
  }

  // SilverGenerals moves
  {
    pieces = ~pinned & board[BB::Type::SILVER_GENERAL] &
             board[BB::Type::ALL_WHITE] & notPromoted;
    moves = moveS(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]) * 2;  // promotions
    moves = moveSE(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]) * 2;  // promotions
    moves = moveSW(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]) * 2;  // promotions
    moves = moveNE(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) +
                     d_popcount(moves[MID] & BOTTOM_RANK) *
                         2 +  // promotion when starting from promotion zone
                     d_popcount(moves[MID] & ~BOTTOM_RANK) +
                     d_popcount(moves[BOTTOM]) * 2;  // promotions
    moves = moveNW(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) +
                     d_popcount(moves[MID] & BOTTOM_RANK) *
                         2 +  // promotion when starting from promotion zone
                     d_popcount(moves[MID] & ~BOTTOM_RANK) +
                     d_popcount(moves[BOTTOM]) * 2;  // promotions
  }

  // GoldGenerals moves
  {
    pieces = ~pinned &
             (board[BB::Type::GOLD_GENERAL] |
              ((board[BB::Type::PAWN] | board[BB::Type::LANCE] |
                board[BB::Type::KNIGHT] | board[BB::Type::SILVER_GENERAL]) &
               board[BB::Type::PROMOTED])) &
             board[BB::Type::ALL_WHITE];
    moves = moveS(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]);
    moves = moveSE(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]);
    moves = moveSW(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]);
    moves = moveE(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]);
    moves = moveW(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]);
    moves = moveN(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]);
  }

  // Lance moves
  {
    pieces = ~pinned & board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] &
             notPromoted;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      moves = GPU::getFileAttacks(square, occupied) &
              ~GPU::getRankMask(squareToRank(square)) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
          d_popcount(moves[BOTTOM] & ~BOTTOM_RANK) * 2 +  // promotions
          d_popcount(moves[BOTTOM] & BOTTOM_RANK);        // forced promotion
    }
  }

  // Bishop moves
  {
    pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
             notPromoted;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      moves = (GPU::getDiagRightAttacks(square, occupied) |
               GPU::getDiagLeftAttacks(square, occupied)) &
              validMoves;
      ourAttacks |= moves;
      if (square >= WHITE_PROMOTION_START) {  // Starting from promotion zone
        numberOfMoves += (d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                          d_popcount(moves[BOTTOM])) *
                         2;
      } else {
        numberOfMoves +=
            d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
            d_popcount(moves[BOTTOM]) * 2;  // end in promotion Zone
      }
    }
  }

  // Rook moves
  {
    pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
             notPromoted;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      moves = (GPU::getRankAttacks(square, occupied) |
               GPU::getFileAttacks(square, occupied)) &
              validMoves;
      ourAttacks |= moves;
      if (square >= WHITE_PROMOTION_START) {  // Starting from promotion zone
        numberOfMoves += (d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                          d_popcount(moves[BOTTOM])) *
                         2;
      } else {
        numberOfMoves +=
            d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
            d_popcount(moves[BOTTOM]) * 2;  // end in promotion Zone
      }
    }
  }

  // Horse moves
  {
    pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
             board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      Bitboard horse = Bitboard(square);
      moves = (GPU::getDiagRightAttacks(square, occupied) |
               GPU::getDiagLeftAttacks(square, occupied) | moveN(horse) |
               moveE(horse) | moveS(horse) | moveW(horse)) &
              validMoves;
      ourAttacks |= moves;
      numberOfMoves += (d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                        d_popcount(moves[BOTTOM]));
    }
  }

  // Dragon moves
  {
    pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
             board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      Bitboard dragon(square);
      moves = (GPU::getRankAttacks(square, occupied) |
               GPU::getFileAttacks(square, occupied) | moveNW(dragon) |
               moveNE(dragon) | moveSE(dragon) | moveSW(dragon)) &
              validMoves;
      ourAttacks |= moves;
      numberOfMoves += (d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                        d_popcount(moves[BOTTOM]));
    }
  }

  // Drop moves
  {
    Bitboard legalDropSpots;
    // Pawns
    if (board.inHand.pieceNumber.WhitePawn > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last rank
      legalDropSpots[BOTTOM] &= ~BOTTOM_RANK;
      // Cannot drop to give checkmate
      pieces = board[BB::Type::KING] & board[BB::Type::ALL_BLACK];
      // All valid enemy king moves
      moves =
          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
          ~ourAttacks & ~board[BB::Type::ALL_BLACK];
      // If there is only one spot pawn cannot block it
      if (d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
              d_popcount(moves[BOTTOM]) ==
          1) {
        legalDropSpots &= ~moveN(moves);
      }
      // Cannot drop on file with other pawn
      Bitboard validFiles;
      for (int fileIdx = 0; fileIdx < BOARD_DIM; fileIdx++) {
        Bitboard file = getFullFile(fileIdx);
        if (!(file & board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
              notPromoted)) {
          validFiles |= file;
        }
      }
      legalDropSpots &= validFiles;
      numberOfMoves += d_popcount(legalDropSpots[TOP]) +
                       d_popcount(legalDropSpots[MID]) +
                       d_popcount(legalDropSpots[BOTTOM]);
    }
    if (board.inHand.pieceNumber.WhiteLance > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last rank
      legalDropSpots[BOTTOM] &= ~BOTTOM_RANK;
      numberOfMoves += d_popcount(legalDropSpots[TOP]) +
                       d_popcount(legalDropSpots[MID]) +
                       d_popcount(legalDropSpots[BOTTOM]);
    }
    if (board.inHand.pieceNumber.WhiteKnight > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last two ranks
      legalDropSpots[BOTTOM] &= TOP_RANK;
      numberOfMoves += d_popcount(legalDropSpots[TOP]) +
                       d_popcount(legalDropSpots[MID]) +
                       d_popcount(legalDropSpots[BOTTOM]);
    }
    legalDropSpots = validMoves & ~occupied;
    numberOfMoves +=
        ((board.inHand.pieceNumber.WhiteSilverGeneral > 0) +
         (board.inHand.pieceNumber.WhiteGoldGeneral > 0) +
         (board.inHand.pieceNumber.WhiteBishop > 0) +
         (board.inHand.pieceNumber.WhiteRook > 0)) *
        (d_popcount(legalDropSpots[TOP]) + d_popcount(legalDropSpots[MID]) +
         d_popcount(legalDropSpots[BOTTOM]));
  }
  if (numberOfMoves == 0) {
    *isMate = true;
    return;
  }

  outMovesOffset[index] = numberOfMoves;
}

__global__ void countBlackMovesKernel(Board* inBoards,
                                      uint32_t inBoardsLength,
                                      Bitboard* outValidMoves,
                                      Bitboard* outAttackedByEnemy,
                                      Bitboard* outPinned,
                                      uint32_t* outMovesOffset,
                                      bool* isMate) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= inBoardsLength)
    return;

  Board board = inBoards[index];
  Bitboard king = board[BB::Type::KING] & board[BB::Type::ALL_BLACK];
  Bitboard notPromoted = ~board[BB::Type::PROMOTED];
  Bitboard checkingPieces, attacked, pieces, moves, slidingChecksPaths, attacks,
      attacksFull, mask, potentialPin, pinned, ourAttacks;
  BitboardIterator iterator;
  Square square;
  size_t numberOfMoves = 0;

  // Non Sliding pieces
  // Pawns
  pieces =
      board.bbs[BB::Type::PAWN] & board.bbs[BB::Type::ALL_WHITE] & notPromoted;
  checkingPieces |= moveN(king) & pieces;
  attacked |= moveS(pieces);
  // Knights
  pieces = board.bbs[BB::Type::KNIGHT] & board.bbs[BB::Type::ALL_WHITE] &
           notPromoted;
  checkingPieces |= moveN(moveNE(king) | moveNW(king)) & pieces;
  attacked |= moveS(moveSE(pieces) | moveSW(pieces));
  // Silver generals
  pieces = board.bbs[BB::Type::SILVER_GENERAL] &
           board.bbs[BB::Type::ALL_WHITE] & notPromoted;
  checkingPieces |= (moveNE(king) | moveN(king) | moveNW(king) | moveSE(king) |
                     moveSW(king)) &
                    pieces;
  attacked |= moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveNE(pieces) |
              moveNW(pieces);
  // gold generals
  pieces = (board[BB::Type::GOLD_GENERAL] |
            ((board[BB::Type::PAWN] | board[BB::Type::KNIGHT] |
              board[BB::Type::SILVER_GENERAL] | board[BB::Type::LANCE]) &
             board[BB::Type::PROMOTED])) &
           board.bbs[BB::Type::ALL_WHITE];
  checkingPieces |= (moveNE(king) | moveN(king) | moveNW(king) | moveE(king) |
                     moveW(king) | moveS(king)) &
                    pieces;
  attacked |= moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveE(pieces) |
              moveW(pieces) | moveN(pieces);
  // Horse (non sliding part)
  pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
           board[BB::Type::PROMOTED];
  checkingPieces |=
      (moveN(king) | moveE(king) | moveS(king) | moveW(king)) & pieces;
  attacked |= moveN(pieces) | moveE(pieces) | moveS(pieces) | moveW(pieces);
  // Dragon (non sldiing part)
  pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
           board[BB::Type::PROMOTED];
  checkingPieces |=
      (moveNW(king) | moveNE(king) | moveSE(king) | moveSW(king)) & pieces;
  attacked |= moveNW(pieces) | moveNE(pieces) | moveSE(pieces) | moveSW(pieces);

  // Sliding pieces
  iterator.Init(king);
  iterator.d_Next();
  Square kingSquare = iterator.GetCurrentSquare();
  Bitboard occupied = board[BB::Type::ALL_BLACK] | board[BB::Type::ALL_WHITE];
  // Lance
  {
    pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] & notPromoted;
    checkingPieces |= GPU::getFileAttacks(kingSquare, occupied) &
                      GPU::getRankMask(squareToRank(kingSquare)) & pieces;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      attacksFull = GPU::getFileAttacks(square, occupied);
      attacked |= attacksFull;
      mask = ~GPU::getRankMask(squareToRank(square));
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::ALL_BLACK];
        attacks = GPU::getFileAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
    }
  }

  // Rook and dragon
  {
    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE];
    checkingPieces |= (GPU::getRankAttacks(kingSquare, occupied) |
                       GPU::getFileAttacks(kingSquare, occupied)) &
                      pieces;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      // Check if king is in check without white pieces
      // We have to check all 4 directions
      // left-right
      attacksFull = GPU::getRankAttacks(square, occupied);
      attacked |= attacksFull;
      mask = GPU::getFileMask(squareToFile(square));
      // left
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & mask;
        attacks = GPU::getRankAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
      // right
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & ~mask;
        attacks = GPU::getRankAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & ~mask;
      }
      // up-down
      attacksFull = GPU::getFileAttacks(square, occupied);
      attacked |= attacksFull;
      mask = GPU::getRankMask(squareToRank(square));
      // up
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & mask;
        attacks = GPU::getFileAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
      // down
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & ~mask;
        attacks = GPU::getFileAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & ~mask;
      }
    }
  }

  // Bishop and horse pins
  {
    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE];
    checkingPieces |= (GPU::getDiagRightAttacks(kingSquare, occupied) |
                       GPU::getDiagLeftAttacks(kingSquare, occupied)) &
                      pieces;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      // Check if king is in check without white pieces
      // We have to check all 4 directions
      // right diag
      attacksFull = GPU::getDiagRightAttacks(square, occupied);
      attacked |= attacksFull;
      mask = ~GPU::getFileMask(squareToFile(square)) &
             GPU::getRankMask(squareToRank(square));
      // SW
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & mask;
        attacks = GPU::getDiagRightAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
      // NE
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & ~mask;
        attacks = GPU::getDiagRightAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & ~mask;
      }
      // left diag
      attacksFull = GPU::getDiagLeftAttacks(square, occupied);
      attacked |= attacksFull;
      mask = GPU::getFileMask(squareToFile(square)) &
             GPU::getRankMask(squareToRank(square));
      // NW
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & mask;
        attacks = GPU::getDiagLeftAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
      // SE
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & ~mask;
        attacks = GPU::getDiagLeftAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & ~mask;
      }
    }
  }

  int numberOfCheckingPieces = d_popcount(checkingPieces[TOP]) +
                               d_popcount(checkingPieces[MID]) +
                               d_popcount(checkingPieces[BOTTOM]);

  // King can always move to non attacked squares
  moves = moveNE(king) | moveN(king) | moveNW(king) | moveE(king) |
          moveW(king) | moveSE(king) | moveS(king) | moveSW(king);
  moves &= ~attacked & ~board[BB::Type::ALL_BLACK];
  numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                   d_popcount(moves[BOTTOM]);
  Bitboard validMoves;
  // If more then one piece is checking the king and king cannot move its mate
  if (numberOfCheckingPieces > 1) {
    if (numberOfMoves == 0) {
      *isMate = true;
      return;
    }
  } else if (numberOfCheckingPieces == 1) {
    // if king is checked by exactly one piece legal moves can also be block
    // sliding check or capture a checking piece
    validMoves = checkingPieces | (slidingChecksPaths & ~king);
  } else if (numberOfCheckingPieces == 0) {
    // If there is no checks all moves are valid (you cannot capture your own
    // piece)
    validMoves = ~board[BB::Type::ALL_BLACK];
  }

  outValidMoves[index] = validMoves;
  outAttackedByEnemy[index] = attacked;
  outPinned[index] = pinned;

  // Pawn moves
  {
    pieces = ~pinned & board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
             notPromoted;
    moves = moveN(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP] & TOP_RANK) +  // forced promotions
                     d_popcount(moves[TOP] & ~TOP_RANK) * 2 +  // promotions
                     d_popcount(moves[MID]) + d_popcount(moves[BOTTOM]);
  }

  // Knight moves
  {
    pieces = ~pinned & board[BB::Type::KNIGHT] & board[BB::Type::ALL_BLACK] &
             notPromoted;
    moves = moveN(moveNE(pieces)) & validMoves;
    ourAttacks |= moves;
    numberOfMoves +=
        d_popcount(moves[TOP] & ~BOTTOM_RANK) +     // forced promotions
        d_popcount(moves[TOP] & BOTTOM_RANK) * 2 +  // promotions
        d_popcount(moves[MID]) + d_popcount(moves[BOTTOM]);
    moves = moveN(moveNW(pieces)) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP] & ~TOP_RANK) +  // forced promotions
                     d_popcount(moves[TOP] & TOP_RANK) * 2 +  // promotions
                     d_popcount(moves[MID]) + d_popcount(moves[BOTTOM]);
  }

  // SilverGenerals moves
  {
    pieces = ~pinned & board[BB::Type::SILVER_GENERAL] &
             board[BB::Type::ALL_BLACK] & notPromoted;
    moves = moveN(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) * 2 +  // promotions
                     d_popcount(moves[MID]) + d_popcount(moves[BOTTOM]);
    moves = moveNE(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) * 2 +  // promotions
                     d_popcount(moves[MID]) + d_popcount(moves[BOTTOM]);
    moves = moveNW(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) * 2 +  // promotions
                     d_popcount(moves[MID]) + d_popcount(moves[BOTTOM]);
    moves = moveSE(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) * 2 +  // promotions
                     d_popcount(moves[MID] & TOP_RANK) *
                         2 +  // promotion when starting from promotion zone
                     d_popcount(moves[MID] & ~TOP_RANK) +
                     d_popcount(moves[BOTTOM]);
    moves = moveSW(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) * 2 +  // promotions
                     d_popcount(moves[MID] & TOP_RANK) *
                         2 +  // promotion when starting from promotion zone
                     d_popcount(moves[MID] & ~TOP_RANK) +
                     d_popcount(moves[BOTTOM]);
  }

  // GoldGenerals moves
  {
    pieces = ~pinned &
             (board[BB::Type::GOLD_GENERAL] |
              ((board[BB::Type::PAWN] | board[BB::Type::LANCE] |
                board[BB::Type::KNIGHT] | board[BB::Type::SILVER_GENERAL]) &
               board[BB::Type::PROMOTED])) &
             board[BB::Type::ALL_BLACK];
    moves = moveN(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]);
    moves = moveNE(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]);
    moves = moveNW(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]);
    moves = moveE(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]);
    moves = moveW(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]);
    moves = moveS(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]);
  }

  // Lance moves
  {
    pieces = ~pinned & board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] &
             notPromoted;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      moves = GPU::getFileAttacks(square, occupied) &
              GPU::getRankMask(squareToRank(square)) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += d_popcount(moves[TOP] & TOP_RANK) +  // forced promotions
                       d_popcount(moves[TOP] & ~TOP_RANK) * 2 +  // promotions
                       d_popcount(moves[MID]) + d_popcount(moves[BOTTOM]);
    }
  }

  // Bishop moves
  {
    pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
             notPromoted;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      moves = (GPU::getDiagRightAttacks(square, occupied) |
               GPU::getDiagLeftAttacks(square, occupied)) &
              validMoves;
      ourAttacks |= moves;
      if (square <= BLACK_PROMOTION_END) {  // Starting from promotion zone
        numberOfMoves += (d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                          d_popcount(moves[BOTTOM])) *
                         2;
      } else {
        numberOfMoves += d_popcount(moves[TOP]) * 2 +  // end in promotion Zone
                         d_popcount(moves[MID]) + d_popcount(moves[BOTTOM]);
      }
    }
  }

  // Rook moves
  {
    pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
             notPromoted;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      moves = (GPU::getRankAttacks(square, occupied) |
               GPU::getFileAttacks(square, occupied)) &
              validMoves;
      ourAttacks |= moves;
      if (square <= BLACK_PROMOTION_END) {  // Starting from promotion zone
        numberOfMoves += (d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                          d_popcount(moves[BOTTOM])) *
                         2;
      } else {
        numberOfMoves += d_popcount(moves[TOP]) * 2 +  // end in promotion Zone
                         d_popcount(moves[MID]) + d_popcount(moves[BOTTOM]);
      }
    }
  }

  // Horse moves
  {
    pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
             board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      Bitboard horse = Bitboard(square);
      moves = (GPU::getDiagRightAttacks(square, occupied) |
               GPU::getDiagLeftAttacks(square, occupied) | moveN(horse) |
               moveE(horse) | moveS(horse) | moveW(horse)) &
              validMoves;
      ourAttacks |= moves;
      numberOfMoves += (d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                        d_popcount(moves[BOTTOM]));
    }
  }

  // Dragon moves
  {
    pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
             board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      Bitboard dragon(square);
      moves = (GPU::getRankAttacks(square, occupied) |
               GPU::getFileAttacks(square, occupied) | moveNW(dragon) |
               moveNE(dragon) | moveSE(dragon) | moveSW(dragon)) &
              validMoves;
      ourAttacks |= moves;
      numberOfMoves += (d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                        d_popcount(moves[BOTTOM]));
    }
  }

  // Drop moves
  {
    Bitboard legalDropSpots;
    // Pawns
    if (board.inHand.pieceNumber.BlackPawn > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last rank
      legalDropSpots[TOP] &= ~TOP_RANK;
      // Cannot drop to give checkmate
      pieces = board[BB::Type::KING] & board[BB::Type::ALL_WHITE];
      // All valid enemy king moves
      moves =
          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
          ~ourAttacks & ~board[BB::Type::ALL_WHITE];
      // If there is only one spot pawn cannot block it
      if (d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
              d_popcount(moves[BOTTOM]) ==
          1) {
        legalDropSpots &= ~moveS(moves);
      }
      // Cannot drop on file with other pawn
      Bitboard validFiles;
      for (int fileIdx = 0; fileIdx < BOARD_DIM; fileIdx++) {
        Bitboard file = getFullFile(fileIdx);
        if (!(file & board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
              notPromoted)) {
          validFiles |= file;
        }
      }
      legalDropSpots &= validFiles;
      numberOfMoves += d_popcount(legalDropSpots[TOP]) +
                       d_popcount(legalDropSpots[MID]) +
                       d_popcount(legalDropSpots[BOTTOM]);
    }
    if (board.inHand.pieceNumber.BlackLance > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last rank
      legalDropSpots[TOP] &= ~TOP_RANK;
      numberOfMoves += d_popcount(legalDropSpots[TOP]) +
                       d_popcount(legalDropSpots[MID]) +
                       d_popcount(legalDropSpots[BOTTOM]);
    }
    if (board.inHand.pieceNumber.BlackKnight > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last two ranks
      legalDropSpots[TOP] &= BOTTOM_RANK;
      numberOfMoves += d_popcount(legalDropSpots[TOP]) +
                       d_popcount(legalDropSpots[MID]) +
                       d_popcount(legalDropSpots[BOTTOM]);
    }
    legalDropSpots = validMoves & ~occupied;
    numberOfMoves +=
        ((board.inHand.pieceNumber.BlackSilverGeneral > 0) +
         (board.inHand.pieceNumber.BlackGoldGeneral > 0) +
         (board.inHand.pieceNumber.BlackBishop > 0) +
         (board.inHand.pieceNumber.BlackRook > 0)) *
        (d_popcount(legalDropSpots[TOP]) + d_popcount(legalDropSpots[MID]) +
         d_popcount(legalDropSpots[BOTTOM]));
  }
  if (numberOfMoves == 0) {
    *isMate = true;
    return;
  }
  outMovesOffset[index] = numberOfMoves;
}

__global__ void generateWhiteMovesKernel(Board* inBoards,
                                         uint32_t inBoardsLength,
                                         Bitboard* inValidMoves,
                                         Bitboard* inAttackedByEnemy,
                                         Bitboard* inPinned,
                                         uint32_t* inMovesOffset,
                                         Move* outMoves,
                                         uint32_t* outMoveToBoardIdx) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= inBoardsLength)
    return;

  Board board = inBoards[index];
  Bitboard validMoves = inValidMoves[index];
  Bitboard notPromoted = ~board[BB::Type::PROMOTED];
  Bitboard pinned = inPinned[index];
  Bitboard pieces, moves, ourAttacks;
  Bitboard occupied = board[BB::Type::ALL_WHITE] | board[BB::Type::ALL_BLACK];
  BitboardIterator movesIterator, iterator;
  Move move;
  uint32_t movesCount = inMovesOffset[index + 1] - inMovesOffset[index];
  uint32_t movesOffset = inMovesOffset[index];
  uint32_t moveNumber = 0;
  // Pawn moves
  {
    pieces = ~pinned & board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
             notPromoted;
    moves = moveS(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + N;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
      // Not when forced promotion
      if (move.to < WHITE_PAWN_LANCE_FORECED_PROMOTION_START) {
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
  }
  // Knight moves
  {
    pieces = ~pinned & board[BB::Type::KNIGHT] & board[BB::Type::ALL_WHITE] &
             notPromoted;
    moves = moveS(moveSE(pieces)) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + N + NW;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
      // Not when forced promotion
      if (move.to < WHITE_HORSE_FORCED_PROMOTION_START) {
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
    moves = moveS(moveSW(pieces)) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + N + NE;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
      // Not when forced promotion
      if (move.to < WHITE_HORSE_FORCED_PROMOTION_START) {
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
  }

  // SilverGenerals moves
  {
    pieces = ~pinned & board[BB::Type::SILVER_GENERAL] &
             board[BB::Type::ALL_WHITE] & notPromoted;
    moves = moveS(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + N;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;

      outMoveToBoardIdx[movesOffset + moveNumber] = index;
      moveNumber++;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
    moves = moveSE(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + NW;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;

      outMoveToBoardIdx[movesOffset + moveNumber] = index;
      moveNumber++;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
    moves = moveSW(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + NE;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;

      outMoveToBoardIdx[movesOffset + moveNumber] = index;
      moveNumber++;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
    moves = moveNE(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + SW;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;

      outMoveToBoardIdx[movesOffset + moveNumber] = index;
      moveNumber++;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START ||
          move.from >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
    moves = moveNW(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + SE;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;

      outMoveToBoardIdx[movesOffset + moveNumber] = index;
      moveNumber++;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START ||
          move.from >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
  }

  // GoldGenerals moves
  {
    pieces = ~pinned &
             (board[BB::Type::GOLD_GENERAL] |
              ((board[BB::Type::PAWN] | board[BB::Type::LANCE] |
                board[BB::Type::KNIGHT] | board[BB::Type::SILVER_GENERAL]) &
               board[BB::Type::PROMOTED])) &
             board[BB::Type::ALL_WHITE];
    moves = moveS(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + N;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;

      outMoveToBoardIdx[movesOffset + moveNumber] = index;
      moveNumber++;
    }
    moves = moveSE(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + NW;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;

      outMoveToBoardIdx[movesOffset + moveNumber] = index;
      moveNumber++;
    }
    moves = moveSW(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + NE;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;

      outMoveToBoardIdx[movesOffset + moveNumber] = index;
      moveNumber++;
    }
    moves = moveE(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + W;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;

      outMoveToBoardIdx[movesOffset + moveNumber] = index;
      moveNumber++;
    }
    moves = moveW(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + E;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;

      outMoveToBoardIdx[movesOffset + moveNumber] = index;
      moveNumber++;
    }
    moves = moveN(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + S;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;

      outMoveToBoardIdx[movesOffset + moveNumber] = index;
      moveNumber++;
    }
  }

  // Lances moves
  {
    pieces = ~pinned & board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] &
             notPromoted;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      move.from = iterator.GetCurrentSquare();
      moves = GPU::getFileAttacks(static_cast<Square>(move.from), occupied) &
              ~GPU::getRankMask(squareToRank(static_cast<Square>(move.from))) &
              validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        // Promotion
        if (move.to >= WHITE_PROMOTION_START) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
        // Not when forced promotion
        if (move.to < WHITE_PAWN_LANCE_FORECED_PROMOTION_START) {
          move.promotion = 0;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
      }
    }
  }

  // Bishop moves
  {
    pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
             notPromoted;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      move.from = iterator.GetCurrentSquare();
      moves =
          (GPU::getDiagRightAttacks(static_cast<Square>(move.from), occupied) |
           GPU::getDiagLeftAttacks(static_cast<Square>(move.from), occupied)) &
          validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
        // Promotion
        if (move.to >= WHITE_PROMOTION_START ||
            move.from >= WHITE_PROMOTION_START) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
      }
    }
  }

  // Rook moves
  {
    pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
             notPromoted;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      move.from = iterator.GetCurrentSquare();
      moves = (GPU::getRankAttacks(static_cast<Square>(move.from), occupied) |
               GPU::getFileAttacks(static_cast<Square>(move.from), occupied)) &
              validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
        // Promotion
        if (move.to >= WHITE_PROMOTION_START ||
            move.from >= WHITE_PROMOTION_START) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
      }
    }
  }
  move.promotion = 0;

  // Horse moves
  {
    pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
             board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      move.from = iterator.GetCurrentSquare();
      Bitboard horse(static_cast<Square>(move.from));
      moves =
          (GPU::getDiagRightAttacks(static_cast<Square>(move.from), occupied) |
           GPU::getDiagLeftAttacks(static_cast<Square>(move.from), occupied) |
           moveN(horse) | moveE(horse) | moveS(horse) | moveW(horse)) &
          validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
  }

  // Dragon moves
  {
    pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
             board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      move.from = iterator.GetCurrentSquare();
      Bitboard dragon(static_cast<Square>(move.from));
      moves =
          (GPU::getRankAttacks(static_cast<Square>(move.from), occupied) |
           GPU::getFileAttacks(static_cast<Square>(move.from), occupied) |
           moveNW(pieces) | moveNE(pieces) | moveSE(pieces) | moveSW(pieces)) &
          validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
  }

  // King moves
  {
    pieces = board[BB::Type::KING] & board[BB::Type::ALL_WHITE];
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      move.from = iterator.GetCurrentSquare();
      moves =
          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
          ~inAttackedByEnemy[index] & ~board[BB::Type::ALL_WHITE];
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
  }

  // Generate Drop moves
  {
    Bitboard legalDropSpots;
    // Pawns
    if (board.inHand.pieceNumber.WhitePawn > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last rank
      legalDropSpots[BOTTOM] &= ~BOTTOM_RANK;
      // Cannot drop to give checkmate
      pieces = board[BB::Type::KING] & board[BB::Type::ALL_BLACK];
      // All valid enemy king moves
      moves =
          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
          ~ourAttacks & ~board[BB::Type::ALL_BLACK];
      // If there is only one spot pawn cannot block it
      if (d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
              d_popcount(moves[BOTTOM]) ==
          1) {
        legalDropSpots &= ~moveN(moves);
      }
      // Cannot drop on file with other pawn
      Bitboard validFiles;
      for (int fileIdx = 0; fileIdx < BOARD_DIM; fileIdx++) {
        Bitboard file = getFullFile(fileIdx);
        if (!(file & board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
              notPromoted)) {
          validFiles |= file;
        }
      }
      legalDropSpots &= validFiles;
      movesIterator.Init(legalDropSpots);
      move.from = WHITE_PAWN_DROP;
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
    if (board.inHand.pieceNumber.WhiteLance > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last rank
      legalDropSpots[BOTTOM] &= ~BOTTOM_RANK;
      movesIterator.Init(legalDropSpots);
      move.from = WHITE_LANCE_DROP;
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
    if (board.inHand.pieceNumber.WhiteKnight > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last two ranks
      legalDropSpots[BOTTOM] &= TOP_RANK;
      movesIterator.Init(legalDropSpots);
      move.from = WHITE_KNIGHT_DROP;
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
    legalDropSpots = validMoves & ~occupied;
    movesIterator.Init(legalDropSpots);
    while (movesIterator.d_Next()) {
      if (board.inHand.pieceNumber.WhiteSilverGeneral > 0) {
        move.from = WHITE_SILVER_GENERAL_DROP;
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
      if (board.inHand.pieceNumber.WhiteGoldGeneral > 0) {
        move.from = WHITE_GOLD_GENERAL_DROP;
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
      if (board.inHand.pieceNumber.WhiteBishop > 0) {
        move.from = WHITE_BISHOP_DROP;
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
      if (board.inHand.pieceNumber.WhiteRook > 0) {
        move.from = WHITE_ROOK_DROP;
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
  }
}

__global__ void generateBlackMovesKernel(Board* inBoards,
                                         uint32_t inBoardsLength,
                                         Bitboard* inValidMoves,
                                         Bitboard* inAttackedByEnemy,
                                         Bitboard* inPinned,
                                         uint32_t* inMovesOffset,
                                         Move* outMoves,
                                         uint32_t* outMoveToBoardIdx) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= inBoardsLength)
    return;

  Board board = inBoards[index];
  Bitboard validMoves = inValidMoves[index];
  Bitboard notPromoted = ~board[BB::Type::PROMOTED];
  Bitboard pinned = inPinned[index];
  Bitboard pieces, moves, ourAttacks;
  Bitboard occupied = board[BB::Type::ALL_WHITE] | board[BB::Type::ALL_BLACK];
  BitboardIterator movesIterator, iterator;
  Move move;
  uint32_t movesCount = inMovesOffset[index + 1] - inMovesOffset[index];
  uint32_t movesOffset = inMovesOffset[index];
  uint32_t moveNumber = 0;
  // Pawn moves
  {
    pieces = ~pinned & board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
             notPromoted;
    moves = moveN(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + S;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
      // Not when forced promotion
      if (move.to > BLACK_PAWN_LANCE_FORCE_PROMOTION_END) {
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
  }
  // Knight moves
  {
    pieces = ~pinned & board[BB::Type::KNIGHT] & board[BB::Type::ALL_BLACK] &
             notPromoted;
    moves = moveN(moveNE(pieces)) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + S + SW;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
      // Not when forced promotion
      if (move.to > BLACK_HORSE_FORCED_PROMOTION_END) {
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
    moves = moveN(moveNW(pieces)) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + S + SE;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
      // Not when forced promotion
      if (move.to > BLACK_HORSE_FORCED_PROMOTION_END) {
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
  }
  // SilverGenerals moves
  {
    pieces = ~pinned & board[BB::Type::SILVER_GENERAL] &
             board[BB::Type::ALL_BLACK] & notPromoted;
    moves = moveN(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + S;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;

      outMoveToBoardIdx[movesOffset + moveNumber] = index;
      moveNumber++;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
    moves = moveNE(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + SW;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;

      outMoveToBoardIdx[movesOffset + moveNumber] = index;
      moveNumber++;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
    moves = moveNW(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + SE;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;

      outMoveToBoardIdx[movesOffset + moveNumber] = index;
      moveNumber++;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
    moves = moveSE(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + NW;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;

      outMoveToBoardIdx[movesOffset + moveNumber] = index;
      moveNumber++;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END || move.from <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
    moves = moveSW(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + NE;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;

      outMoveToBoardIdx[movesOffset + moveNumber] = index;
      moveNumber++;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END || move.from <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
  }
  // GoldGenerals moves
  {
    pieces = ~pinned &
             (board[BB::Type::GOLD_GENERAL] |
              ((board[BB::Type::PAWN] | board[BB::Type::LANCE] |
                board[BB::Type::KNIGHT] | board[BB::Type::SILVER_GENERAL]) &
               board[BB::Type::PROMOTED])) &
             board[BB::Type::ALL_BLACK];
    moves = moveN(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + S;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;

      outMoveToBoardIdx[movesOffset + moveNumber] = index;
      moveNumber++;
    }
    moves = moveNE(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + SW;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;

      outMoveToBoardIdx[movesOffset + moveNumber] = index;
      moveNumber++;
    }
    moves = moveNW(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + SE;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;

      outMoveToBoardIdx[movesOffset + moveNumber] = index;
      moveNumber++;
    }
    moves = moveE(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + W;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;

      outMoveToBoardIdx[movesOffset + moveNumber] = index;
      moveNumber++;
    }
    moves = moveW(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + E;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;

      outMoveToBoardIdx[movesOffset + moveNumber] = index;
      moveNumber++;
    }
    moves = moveS(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + N;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;

      outMoveToBoardIdx[movesOffset + moveNumber] = index;
      moveNumber++;
    }
  }
  // Lances moves
  {
    pieces = ~pinned & board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] &
             notPromoted;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      move.from = iterator.GetCurrentSquare();
      moves = GPU::getFileAttacks(static_cast<Square>(move.from), occupied) &
              GPU::getRankMask(squareToRank(static_cast<Square>(move.from))) &
              validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        // Promotion
        if (move.to <= BLACK_PROMOTION_END) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
        // Not when forced promotion
        if (move.to > BLACK_PAWN_LANCE_FORCE_PROMOTION_END) {
          move.promotion = 0;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
      }
    }
  }
  // Bishop moves
  {
    pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
             notPromoted;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      move.from = iterator.GetCurrentSquare();
      moves =
          (GPU::getDiagRightAttacks(static_cast<Square>(move.from), occupied) |
           GPU::getDiagLeftAttacks(static_cast<Square>(move.from), occupied)) &
          validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
        // Promotion
        if (move.to <= BLACK_PROMOTION_END ||
            move.from <= BLACK_PROMOTION_END) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
      }
    }
  }
  // Rook moves
  {
    pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
             notPromoted;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      move.from = iterator.GetCurrentSquare();
      moves = (GPU::getRankAttacks(static_cast<Square>(move.from), occupied) |
               GPU::getFileAttacks(static_cast<Square>(move.from), occupied)) &
              validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
        // Promotion
        if (move.to <= BLACK_PROMOTION_END ||
            move.from <= BLACK_PROMOTION_END) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;

          outMoveToBoardIdx[movesOffset + moveNumber] = index;
          moveNumber++;
        }
      }
    }
  }
  move.promotion = 0;
  // Horse moves
  {
    pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
             board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      move.from = iterator.GetCurrentSquare();
      Bitboard horse(static_cast<Square>(move.from));
      moves =
          (GPU::getDiagRightAttacks(static_cast<Square>(move.from), occupied) |
           GPU::getDiagLeftAttacks(static_cast<Square>(move.from), occupied) |
           moveN(horse) | moveE(horse) | moveS(horse) | moveW(horse)) &
          validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
  }
  // Dragon moves
  {
    pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
             board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      move.from = iterator.GetCurrentSquare();
      Bitboard dragon(static_cast<Square>(move.from));
      moves =
          (GPU::getRankAttacks(static_cast<Square>(move.from), occupied) |
           GPU::getFileAttacks(static_cast<Square>(move.from), occupied) |
           moveNW(dragon) | moveNE(dragon) | moveSE(dragon) | moveSW(dragon)) &
          validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
  }
  // King moves
  {
    pieces = board[BB::Type::KING] & board[BB::Type::ALL_BLACK];
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      move.from = iterator.GetCurrentSquare();
      moves =
          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
          ~inAttackedByEnemy[index] & ~board[BB::Type::ALL_BLACK];
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
  }
  // Drop moves
  {
    Bitboard legalDropSpots;
    // Pawns
    if (board.inHand.pieceNumber.BlackPawn > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last rank
      legalDropSpots[TOP] &= ~TOP_RANK;
      // Cannot drop to give checkmate
      pieces = board[BB::Type::KING] & board[BB::Type::ALL_WHITE];
      // All valid enemy king moves
      moves =
          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
          ~ourAttacks & ~board[BB::Type::ALL_WHITE];
      // If there is only one spot pawn cannot block it
      if (d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
              d_popcount(moves[BOTTOM]) ==
          1) {
        legalDropSpots &= ~moveS(moves);
      }
      // Cannot drop on file with other pawn
      Bitboard validFiles;
      for (int fileIdx = 0; fileIdx < BOARD_DIM; fileIdx++) {
        Bitboard file = getFullFile(fileIdx);
        if (!(file & board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
              notPromoted)) {
          validFiles |= file;
        }
      }
      legalDropSpots &= validFiles;
      move.from = BLACK_PAWN_DROP;
      movesIterator.Init(legalDropSpots);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
    if (board.inHand.pieceNumber.BlackLance > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last rank
      legalDropSpots[TOP] &= ~TOP_RANK;
      move.from = BLACK_LANCE_DROP;
      movesIterator.Init(legalDropSpots);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
    if (board.inHand.pieceNumber.BlackKnight > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last two ranks
      legalDropSpots[TOP] &= BOTTOM_RANK;
      move.from = BLACK_KNIGHT_DROP;
      movesIterator.Init(legalDropSpots);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
    legalDropSpots = validMoves & ~occupied;
    movesIterator.Init(legalDropSpots);
    while (movesIterator.d_Next()) {
      if (board.inHand.pieceNumber.BlackSilverGeneral > 0) {
        move.from = BLACK_SILVER_GENERAL_DROP;
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
      if (board.inHand.pieceNumber.BlackGoldGeneral > 0) {
        move.from = BLACK_GOLD_GENERAL_DROP;
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
      if (board.inHand.pieceNumber.BlackBishop > 0) {
        move.from = BLACK_BISHOP_DROP;
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
      if (board.inHand.pieceNumber.BlackRook > 0) {
        move.from = BLACK_ROOK_DROP;
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;

        outMoveToBoardIdx[movesOffset + moveNumber] = index;
        moveNumber++;
      }
    }
  }
}

__global__ void generateWhiteBoardsKernel(Move* inMoves,
                                          uint32_t inMovesLength,
                                          Board* inBoards,
                                          uint32_t* moveToBoardIdx,
                                          Board* outBoards) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= inMovesLength)
    return;
  uint64_t one = 1;
  Board board = inBoards[moveToBoardIdx[index]];
  Move move = inMoves[index];
  Region toRegionIdx = squareToRegion(static_cast<Square>(move.to));
  uint32_t toRegion = 1 << (REGION_SIZE - 1 - move.to % REGION_SIZE);
  if (move.from < SQUARE_SIZE) {
    Region fromRegionIdx = squareToRegion(static_cast<Square>(move.from));
    uint32_t fromRegion = 1 << (REGION_SIZE - 1 - move.from % REGION_SIZE);
    for (int i = 0; i < BB::Type::KING; i++) {
      if (board[static_cast<BB::Type>(i)][toRegionIdx] & toRegion) {
        board[static_cast<BB::Type>(i)][toRegionIdx] &= ~toRegion;
        uint64_t addedValue = one << (4 * i);
        board.inHand.value += addedValue;
      }
      if (board[static_cast<BB::Type>(i)][fromRegionIdx] & fromRegion) {
        board[static_cast<BB::Type>(i)][fromRegionIdx] &= ~fromRegion;
        board[static_cast<BB::Type>(i)][toRegionIdx] |= toRegion;
      }
    }
    for (int i = BB::Type::KING; i < BB::Type::SIZE; i++) {
      if (board[static_cast<BB::Type>(i)][toRegionIdx] & toRegion) {
        board[static_cast<BB::Type>(i)][toRegionIdx] &= ~toRegion;
      }
      if (board[static_cast<BB::Type>(i)][fromRegionIdx] & fromRegion) {
        board[static_cast<BB::Type>(i)][fromRegionIdx] &= ~fromRegion;
        board[static_cast<BB::Type>(i)][toRegionIdx] |= toRegion;
      }
    }
    if (move.promotion) {
      board[BB::Type::PROMOTED][toRegionIdx] |= toRegion;
    }
  } else {
    int offset = move.from - WHITE_PAWN_DROP;
    uint64_t addedValue = one << (4 * offset);
    board.inHand.value -= addedValue;
    board[static_cast<BB::Type>(offset % 7)][toRegionIdx] |= toRegion;
    board[static_cast<BB::Type>(BB::Type::ALL_WHITE)][toRegionIdx] |= toRegion;
  }
  outBoards[index] = board;
}

__global__ void generateBlackBoardsKernel(Move* inMoves,
                                          uint32_t inMovesLength,
                                          Board* inBoards,
                                          uint32_t* moveToBoardIdx,
                                          Board* outBoards) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= inMovesLength)
    return;

  uint64_t one = 1;
  Board board = inBoards[moveToBoardIdx[index]];
  Move move = inMoves[index];
  Region toRegionIdx = squareToRegion(static_cast<Square>(move.to));
  uint32_t toRegion = 1 << (REGION_SIZE - 1 - move.to % REGION_SIZE);
  if (move.from < SQUARE_SIZE) {
    Region fromRegionIdx = squareToRegion(static_cast<Square>(move.from));
    uint32_t fromRegion = 1 << (REGION_SIZE - 1 - move.from % REGION_SIZE);
    for (int i = 0; i < BB::Type::KING; i++) {
      if (board[static_cast<BB::Type>(i)][toRegionIdx] & toRegion) {
        board[static_cast<BB::Type>(i)][toRegionIdx] &= ~toRegion;
        uint64_t addedValue = one << (4 * (7 + i));
        board.inHand.value += addedValue;
      }
      if (board[static_cast<BB::Type>(i)][fromRegionIdx] & fromRegion) {
        board[static_cast<BB::Type>(i)][fromRegionIdx] &= ~fromRegion;
        board[static_cast<BB::Type>(i)][toRegionIdx] |= toRegion;
      }
    }
    for (int i = BB::Type::KING; i < BB::Type::SIZE; i++) {
      if (board[static_cast<BB::Type>(i)][toRegionIdx] & toRegion) {
        board[static_cast<BB::Type>(i)][toRegionIdx] &= ~toRegion;
      }
      if (board[static_cast<BB::Type>(i)][fromRegionIdx] & fromRegion) {
        board[static_cast<BB::Type>(i)][fromRegionIdx] &= ~fromRegion;
        board[static_cast<BB::Type>(i)][toRegionIdx] |= toRegion;
      }
    }
    if (move.promotion) {
      board[BB::Type::PROMOTED][toRegionIdx] |= toRegion;
    }
  } else {
    int offset = move.from - WHITE_PAWN_DROP;
    uint64_t addedValue = one << (4 * offset);
    board.inHand.value -= addedValue;
    board[static_cast<BB::Type>(offset % 7)][toRegionIdx] |= toRegion;
    board[static_cast<BB::Type>(BB::Type::ALL_BLACK)][toRegionIdx] |= toRegion;
  }
  outBoards[index] = board;
}

__global__ void evaluateBoardsKernel(Board* inBoards,
                                     uint32_t inBoardsLength,
                                     int16_t* outValues) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= inBoardsLength)
    return;

  Board board = inBoards[index];
  int16_t whitePoints = 0, blackPoints = 0;
  Bitboard pieces;
  // White
  // Pawns
  pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
           ~board[BB::Type::PROMOTED];
  whitePoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PAWN;
  pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
           board[BB::Type::PROMOTED];
  whitePoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_PAWN;
  whitePoints += board.inHand.pieceNumber.WhitePawn * PieceValue::IN_HAND_LANCE;
  // Lances
  pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] &
           ~board[BB::Type::PROMOTED];
  whitePoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::LANCE;
  pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] &
           board[BB::Type::PROMOTED];
  whitePoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_LANCE;
  whitePoints +=
      board.inHand.pieceNumber.WhiteLance * PieceValue::IN_HAND_LANCE;
  // Knights
  pieces = board[BB::Type::KNIGHT] & board[BB::Type::ALL_WHITE] &
           ~board[BB::Type::PROMOTED];
  whitePoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::KNIGHT;
  pieces = board[BB::Type::KNIGHT] & board[BB::Type::ALL_WHITE] &
           board[BB::Type::PROMOTED];
  whitePoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_KNIGHT;
  whitePoints +=
      board.inHand.pieceNumber.WhiteKnight * PieceValue::IN_HAND_KNIGHT;
  // SilverGenerals
  pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_WHITE] &
           ~board[BB::Type::PROMOTED];
  whitePoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::SILVER_GENERAL;
  pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_WHITE] &
           board[BB::Type::PROMOTED];
  whitePoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_SILVER_GENERAL;
  whitePoints += board.inHand.pieceNumber.WhiteSilverGeneral *
                 PieceValue::IN_HAND_SILVER_GENERAL;
  // GoldGenerals
  pieces = board[BB::Type::GOLD_GENERAL] & board[BB::Type::ALL_WHITE] &
           ~board[BB::Type::PROMOTED];
  whitePoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::GOLD_GENERAL;
  whitePoints += board.inHand.pieceNumber.WhiteGoldGeneral *
                 PieceValue::IN_HAND_GOLD_GENERAL;
  // Bishops
  pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
           ~board[BB::Type::PROMOTED];
  whitePoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::BISHOP;
  pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
           board[BB::Type::PROMOTED];
  whitePoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_BISHOP;
  whitePoints +=
      board.inHand.pieceNumber.WhiteBishop * PieceValue::IN_HAND_BISHOP;
  // Rooks
  pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
           ~board[BB::Type::PROMOTED];
  whitePoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::ROOK;
  pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
           board[BB::Type::PROMOTED];
  whitePoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_ROOK;
  whitePoints += board.inHand.pieceNumber.WhiteRook * PieceValue::IN_HAND_ROOK;

  // Black
  // Pawns
  pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
           ~board[BB::Type::PROMOTED];
  blackPoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PAWN;
  pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
           board[BB::Type::PROMOTED];
  blackPoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_PAWN;
  blackPoints += board.inHand.pieceNumber.BlackPawn * PieceValue::IN_HAND_LANCE;
  // Lances
  pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] &
           ~board[BB::Type::PROMOTED];
  blackPoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::LANCE;
  pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] &
           board[BB::Type::PROMOTED];
  blackPoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_LANCE;
  blackPoints +=
      board.inHand.pieceNumber.BlackLance * PieceValue::IN_HAND_LANCE;
  // Knights
  pieces = board[BB::Type::KNIGHT] & board[BB::Type::ALL_BLACK] &
           ~board[BB::Type::PROMOTED];
  blackPoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::KNIGHT;
  pieces = board[BB::Type::KNIGHT] & board[BB::Type::ALL_BLACK] &
           board[BB::Type::PROMOTED];
  blackPoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_KNIGHT;
  blackPoints +=
      board.inHand.pieceNumber.BlackKnight * PieceValue::IN_HAND_KNIGHT;
  // SilverGenerals
  pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_BLACK] &
           ~board[BB::Type::PROMOTED];
  blackPoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::SILVER_GENERAL;
  pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_BLACK] &
           board[BB::Type::PROMOTED];
  blackPoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_SILVER_GENERAL;
  blackPoints += board.inHand.pieceNumber.BlackSilverGeneral *
                 PieceValue::IN_HAND_SILVER_GENERAL;
  // GoldGenerals
  pieces = board[BB::Type::GOLD_GENERAL] & board[BB::Type::ALL_BLACK] &
           ~board[BB::Type::PROMOTED];
  blackPoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::GOLD_GENERAL;
  blackPoints += board.inHand.pieceNumber.BlackGoldGeneral *
                 PieceValue::IN_HAND_GOLD_GENERAL;
  // Bishops
  pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
           ~board[BB::Type::PROMOTED];
  blackPoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::BISHOP;
  pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
           board[BB::Type::PROMOTED];
  blackPoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_BISHOP;
  blackPoints +=
      board.inHand.pieceNumber.BlackBishop * PieceValue::IN_HAND_BISHOP;
  // Rooks
  pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
           ~board[BB::Type::PROMOTED];
  blackPoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::ROOK;
  pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
           board[BB::Type::PROMOTED];
  blackPoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_ROOK;
  blackPoints += board.inHand.pieceNumber.BlackRook * PieceValue::IN_HAND_ROOK;

  outValues[index] = whitePoints - blackPoints;
}

int GPU::countWhiteMoves(thrust::device_ptr<Board> inBoards,
                         uint32_t inBoardsLength,
                         thrust::device_ptr<Bitboard> outValidMoves,
                         thrust::device_ptr<Bitboard> outAttackedByEnemy,
                         thrust::device_ptr<Bitboard> outPinned,
                         thrust::device_ptr<uint32_t> outMovesOffset,
                         thrust::device_ptr<bool> isMate) {
  int numOfBlocks = calculateNumberOfBlocks(inBoardsLength);
  countWhiteMovesKernel<<<numOfBlocks, THREADS_COUNT>>>(
      inBoards.get(), inBoardsLength, outValidMoves.get(),
      outAttackedByEnemy.get(), outPinned.get(), outMovesOffset.get(),
      isMate.get());

  cudaError_t cudaStatus;
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "countWhiteMoves launch failed: %s\n",
            cudaGetErrorString(cudaStatus));
    return -1;
  }
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr,
            "cudaDeviceSynchronize returned error code %d after launching "
            "countWhiteMoves!\n",
            cudaStatus);
    return -1;
  }
  return 0;
}

int GPU::countBlackMoves(thrust::device_ptr<Board> inBoards,
                         uint32_t inBoardsLength,
                         thrust::device_ptr<Bitboard> outValidMoves,
                         thrust::device_ptr<Bitboard> outAttackedByEnemy,
                         thrust::device_ptr<Bitboard> outPinned,
                         thrust::device_ptr<uint32_t> outMovesOffset,
                         thrust::device_ptr<bool> isMate) {
  int numOfBlocks = calculateNumberOfBlocks(inBoardsLength);
  countBlackMovesKernel<<<numOfBlocks, THREADS_COUNT>>>(
      inBoards.get(), inBoardsLength, outValidMoves.get(),
      outAttackedByEnemy.get(), outPinned.get(), outMovesOffset.get(),
      isMate.get());

  cudaError_t cudaStatus;
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "countBlackMoves launch failed: %s\n",
            cudaGetErrorString(cudaStatus));
    return -1;
  }
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr,
            "cudaDeviceSynchronize returned error code %d after launching "
            "countBlackMoves!\n",
            cudaStatus);
    return -1;
  }
  return 0;
}

int GPU::prefixSum(uint32_t* inValues, uint32_t inValuesLength) {
  thrust::exclusive_scan(thrust::device, inValues, inValues + inValuesLength,
                         inValues);
  cudaError_t cudaStatus;
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "countBlackMoves launch failed: %s\n",
            cudaGetErrorString(cudaStatus));
    return -1;
  }
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr,
            "cudaDeviceSynchronize returned error code %d after launching "
            "countBlackMoves!\n",
            cudaStatus);
    return -1;
  }
  return 0;
}

int GPU::generateWhiteMoves(thrust::device_ptr<Board> inBoards,
                            uint32_t inBoardsLength,
                            thrust::device_ptr<Bitboard> inValidMoves,
                            thrust::device_ptr<Bitboard> inAttackedByEnemy,
                            thrust::device_ptr<Bitboard> inPinned,
                            thrust::device_ptr<uint32_t> inMovesOffset,
                            thrust::device_ptr<Move> outMoves,
                            thrust::device_ptr<uint32_t> outMoveToBoardIdx) {
  int numOfBlocks = calculateNumberOfBlocks(inBoardsLength);
  generateWhiteMovesKernel<<<numOfBlocks, THREADS_COUNT>>>(
      inBoards.get(), inBoardsLength, inValidMoves.get(),
      inAttackedByEnemy.get(), inPinned.get(), inMovesOffset.get(),
      outMoves.get(), outMoveToBoardIdx.get());

  cudaError_t cudaStatus;
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "generateWhiteMoves launch failed: %s\n",
            cudaGetErrorString(cudaStatus));
    return -1;
  }
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr,
            "cudaDeviceSynchronize returned error code %d after launching "
            "generateWhiteMoves!\n",
            cudaStatus);
    return -1;
  }
  return 0;
}

int GPU::generateBlackMoves(thrust::device_ptr<Board> inBoards,
                            uint32_t inBoardsLength,
                            thrust::device_ptr<Bitboard> inValidMoves,
                            thrust::device_ptr<Bitboard> inAttackedByEnemy,
                            thrust::device_ptr<Bitboard> inPinned,
                            thrust::device_ptr<uint32_t> inMovesOffset,
                            thrust::device_ptr<Move> outMoves,
                            thrust::device_ptr<uint32_t> outMoveToBoardIdx) {
  int numOfBlocks = calculateNumberOfBlocks(inBoardsLength);
  generateBlackMovesKernel<<<numOfBlocks, THREADS_COUNT>>>(
      inBoards.get(), inBoardsLength, inValidMoves.get(),
      inAttackedByEnemy.get(), inPinned.get(), inMovesOffset.get(),
      outMoves.get(), outMoveToBoardIdx.get());

  cudaError_t cudaStatus;
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "generateBlackMoves launch failed: %s\n",
            cudaGetErrorString(cudaStatus));
    return -1;
  }
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr,
            "cudaDeviceSynchronize returned error code %d after launching "
            "generateBlackMoves!\n",
            cudaStatus);
    return -1;
  }
  return 0;
}

int GPU::generateWhiteBoards(thrust::device_ptr<Move> inMoves,
                             uint32_t inMovesLength,
                             thrust::device_ptr<Board> inBoards,
                             thrust::device_ptr<uint32_t> inMoveToBoardIdx,
                             thrust::device_ptr<Board> outBoards) {
  int numOfBlocks = calculateNumberOfBlocks(inMovesLength);
  generateWhiteBoardsKernel<<<numOfBlocks, THREADS_COUNT>>>(
      inMoves.get(), inMovesLength, inBoards.get(), inMoveToBoardIdx.get(),
      outBoards.get());

  cudaError_t cudaStatus;
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "generateWhiteBoards launch failed: %s\n",
            cudaGetErrorString(cudaStatus));
    return -1;
  }
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr,
            "cudaDeviceSynchronize returned error code %d after launching "
            "generateWhiteBoards!\n",
            cudaStatus);
    return -1;
  }
  return 0;
}

int GPU::generateBlackBoards(thrust::device_ptr<Move> inMoves,
                             uint32_t inMovesLength,
                             thrust::device_ptr<Board> inBoards,
                             thrust::device_ptr<uint32_t> inMoveToBoardIdx,
                             thrust::device_ptr<Board> outBoards) {
  int numOfBlocks = calculateNumberOfBlocks(inMovesLength);
  generateBlackBoardsKernel<<<numOfBlocks, THREADS_COUNT>>>(
      inMoves.get(), inMovesLength, inBoards.get(), inMoveToBoardIdx.get(),
      outBoards.get());

  cudaError_t cudaStatus;
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "generateBlackBoards launch failed: %s\n",
            cudaGetErrorString(cudaStatus));
    return -1;
  }
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr,
            "cudaDeviceSynchronize returned error code %d after launching "
            "generateBlackBoards!\n",
            cudaStatus);
    return -1;
  }
  return 0;
}

int GPU::evaluateBoards(thrust::device_ptr<Board> inBoards,
                        uint32_t inBoardsLength,
                        thrust::device_ptr<int16_t> outValues) {
  int numOfBlocks = calculateNumberOfBlocks(inBoardsLength);
  evaluateBoardsKernel<<<numOfBlocks, THREADS_COUNT>>>(
      inBoards.get(), inBoardsLength, outValues.get());

  cudaError_t cudaStatus;
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "generateBlackBoards launch failed: %s\n",
            cudaGetErrorString(cudaStatus));
    return -1;
  }
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr,
            "cudaDeviceSynchronize returned error code %d after launching "
            "generateBlackBoards!\n",
            cudaStatus);
    return -1;
  }
  return 0;
}

__global__ void countWhiteMovesKernel(uint32_t size,
                                      int16_t movesPerBoard,
                                      Board* startBoard,
                                      Move* inMoves,
                                      uint32_t* outOffsets,
                                      uint32_t* outBitboards) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size)
    return;

  Board board = *startBoard;
  for (int m = 0; m < movesPerBoard; m++) {
    Move move = inMoves[m * size + index];
    makeMove(board, inMoves[m * size + index]);
  }
  Bitboard king = board[BB::Type::KING] & board[BB::Type::ALL_WHITE];
  Bitboard notPromoted = ~board[BB::Type::PROMOTED];
  Bitboard checkingPieces, attacked, pieces, moves, slidingChecksPaths, attacks,
      attacksFull, mask, potentialPin, pinned, ourAttacks;
  BitboardIterator iterator;
  Square square;
  size_t numberOfMoves = 0;

  // Non Sliding pieces
  // Pawns
  pieces =
      board.bbs[BB::Type::PAWN] & board.bbs[BB::Type::ALL_BLACK] & notPromoted;
  checkingPieces |= moveS(king) & pieces;
  attacked |= moveN(pieces);
  // Knights
  pieces = board.bbs[BB::Type::KNIGHT] & board.bbs[BB::Type::ALL_BLACK] &
           notPromoted;
  checkingPieces |= moveS(moveSE(king) | moveSW(king)) & pieces;
  attacked |= moveN(moveNE(pieces) | moveNW(pieces));
  // Silve generals
  pieces = board.bbs[BB::Type::SILVER_GENERAL] &
           board.bbs[BB::Type::ALL_BLACK] & notPromoted;
  checkingPieces |= (moveSE(king) | moveS(king) | moveSW(king) | moveNE(king) |
                     moveNW(king)) &
                    pieces;
  attacked |= moveNE(pieces) | moveN(pieces) | moveNW(pieces) | moveSE(pieces) |
              moveSW(pieces);
  // Gold generals
  pieces = (board[BB::Type::GOLD_GENERAL] |
            ((board[BB::Type::PAWN] | board[BB::Type::KNIGHT] |
              board[BB::Type::SILVER_GENERAL] | board[BB::Type::LANCE]) &
             board[BB::Type::PROMOTED])) &
           board.bbs[BB::Type::ALL_BLACK];
  checkingPieces |= (moveSE(king) | moveS(king) | moveSW(king) | moveE(king) |
                     moveW(king) | moveN(king)) &
                    pieces;
  attacked |= moveNE(pieces) | moveN(pieces) | moveNW(pieces) | moveE(pieces) |
              moveW(pieces) | moveS(pieces);
  // Horse (non sliding part)
  pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
           board[BB::Type::PROMOTED];
  checkingPieces |=
      (moveN(king) | moveE(king) | moveS(king) | moveW(king)) & pieces;
  attacked |= moveN(pieces) | moveE(pieces) | moveS(pieces) | moveW(pieces);
  // Dragon (non sliding part)
  pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
           board[BB::Type::PROMOTED];
  checkingPieces |=
      (moveNW(king) | moveNE(king) | moveSE(king) | moveSW(king)) & pieces;
  attacked |= moveNW(pieces) | moveNE(pieces) | moveSE(pieces) | moveSW(pieces);

  // Sliding pieces
  iterator.Init(king);
  iterator.d_Next();
  Square kingSquare = iterator.GetCurrentSquare();
  Bitboard occupied = board[BB::Type::ALL_BLACK] | board[BB::Type::ALL_WHITE];
  // Lance
  {
    pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] & notPromoted;
    checkingPieces |= GPU::getFileAttacks(kingSquare, occupied) &
                      ~GPU::getRankMask(squareToRank(kingSquare)) & pieces;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      attacksFull = GPU::getFileAttacks(square, occupied);
      attacked |= attacksFull;
      mask = GPU::getRankMask(squareToRank(square));
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::ALL_WHITE];
        attacks = GPU::getFileAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
    }
  }

  // Rook and dragon
  {
    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK];
    checkingPieces |= (GPU::getRankAttacks(kingSquare, occupied) |
                       GPU::getFileAttacks(kingSquare, occupied)) &
                      pieces;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      // Check if king is in check without white pieces
      // We have to check all 4 directions
      // left-right
      attacksFull = GPU::getRankAttacks(square, occupied);
      attacked |= attacksFull;
      mask = GPU::getFileMask(squareToFile(square));
      // left
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & mask;
        attacks = GPU::getRankAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
      // right
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & ~mask;
        attacks = GPU::getRankAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & ~mask;
      }
      // up-down
      attacksFull = GPU::getFileAttacks(square, occupied);
      attacked |= attacksFull;
      mask = GPU::getRankMask(squareToRank(square));
      // up
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & mask;
        attacks = GPU::getFileAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
      // down
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & ~mask;
        attacks = GPU::getFileAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & ~mask;
      }
    }
  }

  // Bishop and horse pins
  {
    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK];
    checkingPieces |= (GPU::getDiagRightAttacks(kingSquare, occupied) |
                       GPU::getDiagLeftAttacks(kingSquare, occupied)) &
                      pieces;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      // Check if king is in check without white pieces
      // We have to check all 4 directions
      // right diag
      attacksFull = GPU::getDiagRightAttacks(square, occupied);
      attacked |= attacksFull;
      mask = ~GPU::getFileMask(squareToFile(square)) &
             GPU::getRankMask(squareToRank(square));
      // SW
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & mask;
        attacks = GPU::getDiagRightAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
      // NE
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & ~mask;
        attacks = GPU::getDiagRightAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & ~mask;
      }
      // left diag
      attacksFull = GPU::getDiagLeftAttacks(square, occupied);
      attacked |= attacksFull;
      mask = GPU::getFileMask(squareToFile(square)) &
             GPU::getRankMask(squareToRank(square));
      // NW
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & mask;
        attacks = GPU::getDiagLeftAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
      // SE
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_WHITE] & ~mask;
        attacks = GPU::getDiagLeftAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & ~mask;
      }
    }
  }

  int numberOfCheckingPieces = d_popcount(checkingPieces[TOP]) +
                               d_popcount(checkingPieces[MID]) +
                               d_popcount(checkingPieces[BOTTOM]);

  // King can always move to non attacked squares
  moves = moveNE(king) | moveN(king) | moveNW(king) | moveE(king) |
          moveW(king) | moveSE(king) | moveS(king) | moveSW(king);
  moves &= ~attacked & ~board[BB::Type::ALL_WHITE];
  numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                   d_popcount(moves[BOTTOM]);

  Bitboard validMoves;
  // If more then one piece is checking the king and king cannot move its mate
  if (numberOfCheckingPieces == 1) {
    // if king is checked by exactly one piece legal moves can also be block
    // sliding check or capture a checking piece
    validMoves = checkingPieces | (slidingChecksPaths & ~king);
  } else if (numberOfCheckingPieces == 0) {
    // If there is no checks all moves are valid (you cannot capture your own
    // piece)
    validMoves = ~board[BB::Type::ALL_WHITE];
  }

  outBitboards[index] = validMoves[TOP];
  outBitboards[size + index] = validMoves[MID];
  outBitboards[2 * size + index] = validMoves[BOTTOM];
  outBitboards[3 * size + index] = attacked[TOP];
  outBitboards[4 * size + index] = attacked[MID];
  outBitboards[5 * size + index] = attacked[BOTTOM];
  outBitboards[6 * size + index] = pinned[TOP];
  outBitboards[7 * size + index] = pinned[MID];
  outBitboards[8 * size + index] = pinned[BOTTOM];

  // Pawn moves
  {
    pieces = ~pinned & board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
             notPromoted;
    moves = moveS(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves +=
        d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
        d_popcount(moves[BOTTOM] & ~BOTTOM_RANK) * 2 +  // promotions
        d_popcount(moves[BOTTOM] & BOTTOM_RANK);        // forced promotion
  }

  // Knight moves
  {
    pieces = ~pinned & board[BB::Type::KNIGHT] & board[BB::Type::ALL_WHITE] &
             notPromoted;
    moves = moveS(moveSE(pieces)) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM] & TOP_RANK) * 2 +  // promotions
                     d_popcount(moves[BOTTOM] & ~TOP_RANK);  // forced promotion
    moves = moveS(moveSW(pieces)) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM] & TOP_RANK) * 2 +  // promotions
                     d_popcount(moves[BOTTOM] & ~TOP_RANK);  // forced promotion
  }

  // SilverGenerals moves
  {
    pieces = ~pinned & board[BB::Type::SILVER_GENERAL] &
             board[BB::Type::ALL_WHITE] & notPromoted;
    moves = moveS(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]) * 2;  // promotions
    moves = moveSE(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]) * 2;  // promotions
    moves = moveSW(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]) * 2;  // promotions
    moves = moveNE(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) +
                     d_popcount(moves[MID] & BOTTOM_RANK) *
                         2 +  // promotion when starting from promotion zone
                     d_popcount(moves[MID] & ~BOTTOM_RANK) +
                     d_popcount(moves[BOTTOM]) * 2;  // promotions
    moves = moveNW(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) +
                     d_popcount(moves[MID] & BOTTOM_RANK) *
                         2 +  // promotion when starting from promotion zone
                     d_popcount(moves[MID] & ~BOTTOM_RANK) +
                     d_popcount(moves[BOTTOM]) * 2;  // promotions
  }

  // GoldGenerals moves
  {
    pieces = ~pinned &
             (board[BB::Type::GOLD_GENERAL] |
              ((board[BB::Type::PAWN] | board[BB::Type::LANCE] |
                board[BB::Type::KNIGHT] | board[BB::Type::SILVER_GENERAL]) &
               board[BB::Type::PROMOTED])) &
             board[BB::Type::ALL_WHITE];
    moves = moveS(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]);
    moves = moveSE(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]);
    moves = moveSW(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]);
    moves = moveE(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]);
    moves = moveW(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]);
    moves = moveN(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]);
  }

  // Lance moves
  {
    pieces = ~pinned & board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] &
             notPromoted;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      moves = GPU::getFileAttacks(square, occupied) &
              ~GPU::getRankMask(squareToRank(square)) & validMoves;
      ourAttacks |= moves;
      numberOfMoves +=
          d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
          d_popcount(moves[BOTTOM] & ~BOTTOM_RANK) * 2 +  // promotions
          d_popcount(moves[BOTTOM] & BOTTOM_RANK);        // forced promotion
    }
  }

  // Bishop moves
  {
    pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
             notPromoted;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      moves = (GPU::getDiagRightAttacks(square, occupied) |
               GPU::getDiagLeftAttacks(square, occupied)) &
              validMoves;
      ourAttacks |= moves;
      if (square >= WHITE_PROMOTION_START) {  // Starting from promotion zone
        numberOfMoves += (d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                          d_popcount(moves[BOTTOM])) *
                         2;
      } else {
        numberOfMoves +=
            d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
            d_popcount(moves[BOTTOM]) * 2;  // end in promotion Zone
      }
    }
  }

  // Rook moves
  {
    pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
             notPromoted;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      moves = (GPU::getRankAttacks(square, occupied) |
               GPU::getFileAttacks(square, occupied)) &
              validMoves;
      ourAttacks |= moves;
      if (square >= WHITE_PROMOTION_START) {  // Starting from promotion zone
        numberOfMoves += (d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                          d_popcount(moves[BOTTOM])) *
                         2;
      } else {
        numberOfMoves +=
            d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
            d_popcount(moves[BOTTOM]) * 2;  // end in promotion Zone
      }
    }
  }

  // Horse moves
  {
    pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
             board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      Bitboard horse = Bitboard(square);
      moves = (GPU::getDiagRightAttacks(square, occupied) |
               GPU::getDiagLeftAttacks(square, occupied) | moveN(horse) |
               moveE(horse) | moveS(horse) | moveW(horse)) &
              validMoves;
      ourAttacks |= moves;
      numberOfMoves += (d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                        d_popcount(moves[BOTTOM]));
    }
  }

  // Dragon moves
  {
    pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
             board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      Bitboard dragon(square);
      moves = (GPU::getRankAttacks(square, occupied) |
               GPU::getFileAttacks(square, occupied) | moveNW(dragon) |
               moveNE(dragon) | moveSE(dragon) | moveSW(dragon)) &
              validMoves;
      ourAttacks |= moves;
      numberOfMoves += (d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                        d_popcount(moves[BOTTOM]));
    }
  }

  // Drop moves
  {
    Bitboard legalDropSpots;
    // Pawns
    if (board.inHand.pieceNumber.WhitePawn > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last rank
      legalDropSpots[BOTTOM] &= ~BOTTOM_RANK;
      // Cannot drop to give checkmate
      pieces = board[BB::Type::KING] & board[BB::Type::ALL_BLACK];
      // All valid enemy king moves
      moves =
          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
          ~ourAttacks & ~board[BB::Type::ALL_BLACK];
      // If there is only one spot pawn cannot block it
      if (d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
              d_popcount(moves[BOTTOM]) ==
          1) {
        legalDropSpots &= ~moveN(moves);
      }
      // Cannot drop on file with other pawn
      Bitboard validFiles;
      for (int fileIdx = 0; fileIdx < BOARD_DIM; fileIdx++) {
        Bitboard file = getFullFile(fileIdx);
        if (!(file & board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
              notPromoted)) {
          validFiles |= file;
        }
      }
      legalDropSpots &= validFiles;
      numberOfMoves += d_popcount(legalDropSpots[TOP]) +
                       d_popcount(legalDropSpots[MID]) +
                       d_popcount(legalDropSpots[BOTTOM]);
    }
    if (board.inHand.pieceNumber.WhiteLance > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last rank
      legalDropSpots[BOTTOM] &= ~BOTTOM_RANK;
      numberOfMoves += d_popcount(legalDropSpots[TOP]) +
                       d_popcount(legalDropSpots[MID]) +
                       d_popcount(legalDropSpots[BOTTOM]);
    }
    if (board.inHand.pieceNumber.WhiteKnight > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last two ranks
      legalDropSpots[BOTTOM] &= TOP_RANK;
      numberOfMoves += d_popcount(legalDropSpots[TOP]) +
                       d_popcount(legalDropSpots[MID]) +
                       d_popcount(legalDropSpots[BOTTOM]);
    }
    legalDropSpots = validMoves & ~occupied;
    numberOfMoves +=
        ((board.inHand.pieceNumber.WhiteSilverGeneral > 0) +
         (board.inHand.pieceNumber.WhiteGoldGeneral > 0) +
         (board.inHand.pieceNumber.WhiteBishop > 0) +
         (board.inHand.pieceNumber.WhiteRook > 0)) *
        (d_popcount(legalDropSpots[TOP]) + d_popcount(legalDropSpots[MID]) +
         d_popcount(legalDropSpots[BOTTOM]));
  }
  outOffsets[index] = numberOfMoves;
}

__global__ void countBlackMovesKernel(uint32_t size,
                                      int16_t movesPerBoard,
                                      Board* startBoard,
                                      Move* inMoves,
                                      uint32_t* outOffsets,
                                      uint32_t* outBitboards) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size)
    return;
  Board board = *startBoard;
  for (int m = 0; m < movesPerBoard; m++) {
    makeMove(board, inMoves[m * size + index]);
  }
  Bitboard king = board[BB::Type::KING] & board[BB::Type::ALL_BLACK];
  Bitboard notPromoted = ~board[BB::Type::PROMOTED];
  Bitboard checkingPieces, attacked, pieces, moves, slidingChecksPaths, attacks,
      attacksFull, mask, potentialPin, pinned, ourAttacks;
  BitboardIterator iterator;
  Square square;
  size_t numberOfMoves = 0;

  // Non Sliding pieces
  // Pawns
  pieces =
      board.bbs[BB::Type::PAWN] & board.bbs[BB::Type::ALL_WHITE] & notPromoted;
  checkingPieces |= moveN(king) & pieces;
  attacked |= moveS(pieces);
  // Knights
  pieces = board.bbs[BB::Type::KNIGHT] & board.bbs[BB::Type::ALL_WHITE] &
           notPromoted;
  checkingPieces |= moveN(moveNE(king) | moveNW(king)) & pieces;
  attacked |= moveS(moveSE(pieces) | moveSW(pieces));
  // Silver generals
  pieces = board.bbs[BB::Type::SILVER_GENERAL] &
           board.bbs[BB::Type::ALL_WHITE] & notPromoted;
  checkingPieces |= (moveNE(king) | moveN(king) | moveNW(king) | moveSE(king) |
                     moveSW(king)) &
                    pieces;
  attacked |= moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveNE(pieces) |
              moveNW(pieces);
  // gold generals
  pieces = (board[BB::Type::GOLD_GENERAL] |
            ((board[BB::Type::PAWN] | board[BB::Type::KNIGHT] |
              board[BB::Type::SILVER_GENERAL] | board[BB::Type::LANCE]) &
             board[BB::Type::PROMOTED])) &
           board.bbs[BB::Type::ALL_WHITE];
  checkingPieces |= (moveNE(king) | moveN(king) | moveNW(king) | moveE(king) |
                     moveW(king) | moveS(king)) &
                    pieces;
  attacked |= moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveE(pieces) |
              moveW(pieces) | moveN(pieces);
  // Horse (non sliding part)
  pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
           board[BB::Type::PROMOTED];
  checkingPieces |=
      (moveN(king) | moveE(king) | moveS(king) | moveW(king)) & pieces;
  attacked |= moveN(pieces) | moveE(pieces) | moveS(pieces) | moveW(pieces);
  // Dragon (non sldiing part)
  pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
           board[BB::Type::PROMOTED];
  checkingPieces |=
      (moveNW(king) | moveNE(king) | moveSE(king) | moveSW(king)) & pieces;
  attacked |= moveNW(pieces) | moveNE(pieces) | moveSE(pieces) | moveSW(pieces);
  // Sliding pieces
  iterator.Init(king);
  iterator.d_Next();
  Square kingSquare = iterator.GetCurrentSquare();
  Bitboard occupied = board[BB::Type::ALL_BLACK] | board[BB::Type::ALL_WHITE];
  // Lance
  {
    pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] & notPromoted;
    checkingPieces |= GPU::getFileAttacks(kingSquare, occupied) &
                      GPU::getRankMask(squareToRank(kingSquare)) & pieces;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      attacksFull = GPU::getFileAttacks(square, occupied);
      attacked |= attacksFull;
      mask = ~GPU::getRankMask(squareToRank(square));
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::ALL_BLACK];
        attacks = GPU::getFileAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
    }
  }

  // Rook and dragon
  {
    pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE];
    checkingPieces |= (GPU::getRankAttacks(kingSquare, occupied) |
                       GPU::getFileAttacks(kingSquare, occupied)) &
                      pieces;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      // Check if king is in check without white pieces
      // We have to check all 4 directions
      // left-right
      attacksFull = GPU::getRankAttacks(square, occupied);
      attacked |= attacksFull;
      mask = GPU::getFileMask(squareToFile(square));
      // left
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & mask;
        attacks = GPU::getRankAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
      // right
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & ~mask;
        attacks = GPU::getRankAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & ~mask;
      }
      // up-down
      attacksFull = GPU::getFileAttacks(square, occupied);
      attacked |= attacksFull;
      mask = GPU::getRankMask(squareToRank(square));
      // up
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & mask;
        attacks = GPU::getFileAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
      // down
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & ~mask;
        attacks = GPU::getFileAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & ~mask;
      }
    }
  }

  // Bishop and horse pins
  {
    pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE];
    checkingPieces |= (GPU::getDiagRightAttacks(kingSquare, occupied) |
                       GPU::getDiagLeftAttacks(kingSquare, occupied)) &
                      pieces;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      // Check if king is in check without white pieces
      // We have to check all 4 directions
      // right diag
      attacksFull = GPU::getDiagRightAttacks(square, occupied);
      attacked |= attacksFull;
      mask = ~GPU::getFileMask(squareToFile(square)) &
             GPU::getRankMask(squareToRank(square));
      // SW
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & mask;
        attacks = GPU::getDiagRightAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
      // NE
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & ~mask;
        attacks = GPU::getDiagRightAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & ~mask;
      }
      // left diag
      attacksFull = GPU::getDiagLeftAttacks(square, occupied);
      attacked |= attacksFull;
      mask = GPU::getFileMask(squareToFile(square)) &
             GPU::getRankMask(squareToRank(square));
      // NW
      if (!(attacksFull & king & mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & mask;
        attacks = GPU::getDiagLeftAttacks(square, occupied & ~potentialPin);
        if (attacks & king & mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & mask;
      }
      // SE
      if (!(attacksFull & king & ~mask)) {
        potentialPin = attacksFull & board[BB::Type::ALL_BLACK] & ~mask;
        attacks = GPU::getDiagLeftAttacks(square, occupied & ~potentialPin);
        if (attacks & king & ~mask) {
          pinned |= potentialPin;
        }
      } else {
        slidingChecksPaths |= attacksFull & ~mask;
      }
    }
  }

  int numberOfCheckingPieces = d_popcount(checkingPieces[TOP]) +
                               d_popcount(checkingPieces[MID]) +
                               d_popcount(checkingPieces[BOTTOM]);

  // King can always move to non attacked squares
  moves = moveNE(king) | moveN(king) | moveNW(king) | moveE(king) |
          moveW(king) | moveSE(king) | moveS(king) | moveSW(king);
  moves &= ~attacked & ~board[BB::Type::ALL_BLACK];
  numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                   d_popcount(moves[BOTTOM]);
  Bitboard validMoves;
  if (numberOfCheckingPieces == 1) {
    // if king is checked by exactly one piece legal moves can also be block
    // sliding check or capture a checking piece
    validMoves = checkingPieces | (slidingChecksPaths & ~king);
  } else if (numberOfCheckingPieces == 0) {
    // If there is no checks all moves are valid (you cannot capture your own
    // piece)
    validMoves = ~board[BB::Type::ALL_BLACK];
  }
  outBitboards[index] = validMoves[TOP];
  outBitboards[size + index] = validMoves[MID];
  outBitboards[2 * size + index] = validMoves[BOTTOM];
  outBitboards[3 * size + index] = attacked[TOP];
  outBitboards[4 * size + index] = attacked[MID];
  outBitboards[5 * size + index] = attacked[BOTTOM];
  outBitboards[6 * size + index] = pinned[TOP];
  outBitboards[7 * size + index] = pinned[MID];
  outBitboards[8 * size + index] = pinned[BOTTOM];
  // Pawn moves
  {
    pieces = ~pinned & board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
             notPromoted;
    moves = moveN(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP] & TOP_RANK) +  // forced promotions
                     d_popcount(moves[TOP] & ~TOP_RANK) * 2 +  // promotions
                     d_popcount(moves[MID]) + d_popcount(moves[BOTTOM]);
  }

  // Knight moves
  {
    pieces = ~pinned & board[BB::Type::KNIGHT] & board[BB::Type::ALL_BLACK] &
             notPromoted;
    moves = moveN(moveNE(pieces)) & validMoves;
    ourAttacks |= moves;
    numberOfMoves +=
        d_popcount(moves[TOP] & ~BOTTOM_RANK) +     // forced promotions
        d_popcount(moves[TOP] & BOTTOM_RANK) * 2 +  // promotions
        d_popcount(moves[MID]) + d_popcount(moves[BOTTOM]);
    moves = moveN(moveNW(pieces)) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP] & ~TOP_RANK) +  // forced promotions
                     d_popcount(moves[TOP] & TOP_RANK) * 2 +  // promotions
                     d_popcount(moves[MID]) + d_popcount(moves[BOTTOM]);
  }

  // SilverGenerals moves
  {
    pieces = ~pinned & board[BB::Type::SILVER_GENERAL] &
             board[BB::Type::ALL_BLACK] & notPromoted;
    moves = moveN(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) * 2 +  // promotions
                     d_popcount(moves[MID]) + d_popcount(moves[BOTTOM]);
    moves = moveNE(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) * 2 +  // promotions
                     d_popcount(moves[MID]) + d_popcount(moves[BOTTOM]);
    moves = moveNW(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) * 2 +  // promotions
                     d_popcount(moves[MID]) + d_popcount(moves[BOTTOM]);
    moves = moveSE(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) * 2 +  // promotions
                     d_popcount(moves[MID] & TOP_RANK) *
                         2 +  // promotion when starting from promotion zone
                     d_popcount(moves[MID] & ~TOP_RANK) +
                     d_popcount(moves[BOTTOM]);
    moves = moveSW(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) * 2 +  // promotions
                     d_popcount(moves[MID] & TOP_RANK) *
                         2 +  // promotion when starting from promotion zone
                     d_popcount(moves[MID] & ~TOP_RANK) +
                     d_popcount(moves[BOTTOM]);
  }

  // GoldGenerals moves
  {
    pieces = ~pinned &
             (board[BB::Type::GOLD_GENERAL] |
              ((board[BB::Type::PAWN] | board[BB::Type::LANCE] |
                board[BB::Type::KNIGHT] | board[BB::Type::SILVER_GENERAL]) &
               board[BB::Type::PROMOTED])) &
             board[BB::Type::ALL_BLACK];
    moves = moveN(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]);
    moves = moveNE(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]);
    moves = moveNW(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]);
    moves = moveE(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]);
    moves = moveW(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]);
    moves = moveS(pieces) & validMoves;
    ourAttacks |= moves;
    numberOfMoves += d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                     d_popcount(moves[BOTTOM]);
  }

  // Lance moves
  {
    pieces = ~pinned & board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] &
             notPromoted;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      moves = GPU::getFileAttacks(square, occupied) &
              GPU::getRankMask(squareToRank(square)) & validMoves;
      ourAttacks |= moves;
      numberOfMoves += d_popcount(moves[TOP] & TOP_RANK) +  // forced promotions
                       d_popcount(moves[TOP] & ~TOP_RANK) * 2 +  // promotions
                       d_popcount(moves[MID]) + d_popcount(moves[BOTTOM]);
    }
  }

  // Bishop moves
  {
    pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
             notPromoted;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      moves = (GPU::getDiagRightAttacks(square, occupied) |
               GPU::getDiagLeftAttacks(square, occupied)) &
              validMoves;
      ourAttacks |= moves;
      if (square <= BLACK_PROMOTION_END) {  // Starting from promotion zone
        numberOfMoves += (d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                          d_popcount(moves[BOTTOM])) *
                         2;
      } else {
        numberOfMoves += d_popcount(moves[TOP]) * 2 +  // end in promotion Zone
                         d_popcount(moves[MID]) + d_popcount(moves[BOTTOM]);
      }
    }
  }

  // Rook moves
  {
    pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
             notPromoted;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      moves = (GPU::getRankAttacks(square, occupied) |
               GPU::getFileAttacks(square, occupied)) &
              validMoves;
      ourAttacks |= moves;
      if (square <= BLACK_PROMOTION_END) {  // Starting from promotion zone
        numberOfMoves += (d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                          d_popcount(moves[BOTTOM])) *
                         2;
      } else {
        numberOfMoves += d_popcount(moves[TOP]) * 2 +  // end in promotion Zone
                         d_popcount(moves[MID]) + d_popcount(moves[BOTTOM]);
      }
    }
  }

  // Horse moves
  {
    pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
             board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      Bitboard horse = Bitboard(square);
      moves = (GPU::getDiagRightAttacks(square, occupied) |
               GPU::getDiagLeftAttacks(square, occupied) | moveN(horse) |
               moveE(horse) | moveS(horse) | moveW(horse)) &
              validMoves;
      ourAttacks |= moves;
      numberOfMoves += (d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                        d_popcount(moves[BOTTOM]));
    }
  }

  // Dragon moves
  {
    pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
             board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      square = iterator.GetCurrentSquare();
      Bitboard dragon(square);
      moves = (GPU::getRankAttacks(square, occupied) |
               GPU::getFileAttacks(square, occupied) | moveNW(dragon) |
               moveNE(dragon) | moveSE(dragon) | moveSW(dragon)) &
              validMoves;
      ourAttacks |= moves;
      numberOfMoves += (d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
                        d_popcount(moves[BOTTOM]));
    }
  }

  // Drop moves
  {
    Bitboard legalDropSpots;
    // Pawns
    if (board.inHand.pieceNumber.BlackPawn > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last rank
      legalDropSpots[TOP] &= ~TOP_RANK;
      // Cannot drop to give checkmate
      pieces = board[BB::Type::KING] & board[BB::Type::ALL_WHITE];
      // All valid enemy king moves
      moves =
          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
          ~ourAttacks & ~board[BB::Type::ALL_WHITE];
      // If there is only one spot pawn cannot block it
      if (d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
              d_popcount(moves[BOTTOM]) ==
          1) {
        legalDropSpots &= ~moveS(moves);
      }
      // Cannot drop on file with other pawn
      Bitboard validFiles;
      for (int fileIdx = 0; fileIdx < BOARD_DIM; fileIdx++) {
        Bitboard file = getFullFile(fileIdx);
        if (!(file & board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
              notPromoted)) {
          validFiles |= file;
        }
      }
      legalDropSpots &= validFiles;
      numberOfMoves += d_popcount(legalDropSpots[TOP]) +
                       d_popcount(legalDropSpots[MID]) +
                       d_popcount(legalDropSpots[BOTTOM]);
    }
    if (board.inHand.pieceNumber.BlackLance > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last rank
      legalDropSpots[TOP] &= ~TOP_RANK;
      numberOfMoves += d_popcount(legalDropSpots[TOP]) +
                       d_popcount(legalDropSpots[MID]) +
                       d_popcount(legalDropSpots[BOTTOM]);
    }
    if (board.inHand.pieceNumber.BlackKnight > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last two ranks
      legalDropSpots[TOP] &= BOTTOM_RANK;
      numberOfMoves += d_popcount(legalDropSpots[TOP]) +
                       d_popcount(legalDropSpots[MID]) +
                       d_popcount(legalDropSpots[BOTTOM]);
    }
    legalDropSpots = validMoves & ~occupied;
    numberOfMoves +=
        ((board.inHand.pieceNumber.BlackSilverGeneral > 0) +
         (board.inHand.pieceNumber.BlackGoldGeneral > 0) +
         (board.inHand.pieceNumber.BlackBishop > 0) +
         (board.inHand.pieceNumber.BlackRook > 0)) *
        (d_popcount(legalDropSpots[TOP]) + d_popcount(legalDropSpots[MID]) +
         d_popcount(legalDropSpots[BOTTOM]));
  }
  outOffsets[index] = numberOfMoves;
}

__global__ void generateWhiteMovesKernel(uint32_t size,
                                         int16_t movesPerBoard,
                                         Board* startBoard,
                                         Move* inMoves,
                                         uint32_t* inOffsets,
                                         uint32_t* inBitboards,
                                         Move* outMoves) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size)
    return;

  Board board = *startBoard;
  uint32_t movesOffset = inOffsets[index];
  uint32_t numberOfMoves = inOffsets[index + 1] - movesOffset;
  uint32_t allMovesSize = inOffsets[size] - inOffsets[0];
  for (int m = 0; m < movesPerBoard; m++) {
    Move move = inMoves[m * size + index];
    makeMove(board, move);
    for (int i = 0; i < numberOfMoves; i++) {
      outMoves[movesOffset + i] = move;
    }
    movesOffset += allMovesSize;
  }
  Bitboard validMoves, pinned, attackedByEnemy;
  validMoves[TOP] = inBitboards[index];
  validMoves[MID] = inBitboards[size + index];
  validMoves[BOTTOM] = inBitboards[2 * size + index];
  attackedByEnemy[TOP] = inBitboards[3 * size + index];
  attackedByEnemy[MID] = inBitboards[4 * size + index];
  attackedByEnemy[BOTTOM] = inBitboards[5 * size + index];
  pinned[TOP] = inBitboards[6 * size + index];
  pinned[MID] = inBitboards[7 * size + index];
  pinned[BOTTOM] = inBitboards[8 * size + index];
  Bitboard notPromoted = ~board[BB::Type::PROMOTED];
  Bitboard pieces, moves, ourAttacks;
  Bitboard occupied = board[BB::Type::ALL_WHITE] | board[BB::Type::ALL_BLACK];
  BitboardIterator movesIterator, iterator;
  Move move;
  uint32_t moveNumber = 0;
  // Pawn moves
  {
    pieces = ~pinned & board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
             notPromoted;
    moves = moveS(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + N;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
      // Not when forced promotion
      if (move.to < WHITE_PAWN_LANCE_FORECED_PROMOTION_START) {
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
  }
  // Knight moves
  {
    pieces = ~pinned & board[BB::Type::KNIGHT] & board[BB::Type::ALL_WHITE] &
             notPromoted;
    moves = moveS(moveSE(pieces)) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + N + NW;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
      // Not when forced promotion
      if (move.to < WHITE_HORSE_FORCED_PROMOTION_START) {
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
    moves = moveS(moveSW(pieces)) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + N + NE;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
      // Not when forced promotion
      if (move.to < WHITE_HORSE_FORCED_PROMOTION_START) {
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
  }

  // SilverGenerals moves
  {
    pieces = ~pinned & board[BB::Type::SILVER_GENERAL] &
             board[BB::Type::ALL_WHITE] & notPromoted;
    moves = moveS(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + N;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;
      moveNumber++;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
    moves = moveSE(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + NW;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;
      moveNumber++;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
    moves = moveSW(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + NE;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;
      moveNumber++;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
    moves = moveNE(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + SW;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;
      moveNumber++;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START ||
          move.from >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
    moves = moveNW(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + SE;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;
      moveNumber++;
      // Promotion
      if (move.to >= WHITE_PROMOTION_START ||
          move.from >= WHITE_PROMOTION_START) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
  }

  // GoldGenerals moves
  {
    pieces = ~pinned &
             (board[BB::Type::GOLD_GENERAL] |
              ((board[BB::Type::PAWN] | board[BB::Type::LANCE] |
                board[BB::Type::KNIGHT] | board[BB::Type::SILVER_GENERAL]) &
               board[BB::Type::PROMOTED])) &
             board[BB::Type::ALL_WHITE];
    moves = moveS(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + N;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;
      moveNumber++;
    }
    moves = moveSE(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + NW;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;
      moveNumber++;
    }
    moves = moveSW(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + NE;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;
      moveNumber++;
    }
    moves = moveE(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + W;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;
      moveNumber++;
    }
    moves = moveW(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + E;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;
      moveNumber++;
    }
    moves = moveN(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + S;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;
      moveNumber++;
    }
  }

  // Lances moves
  {
    pieces = ~pinned & board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] &
             notPromoted;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      move.from = iterator.GetCurrentSquare();
      moves = GPU::getFileAttacks(static_cast<Square>(move.from), occupied) &
              ~GPU::getRankMask(squareToRank(static_cast<Square>(move.from))) &
              validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        // Promotion
        if (move.to >= WHITE_PROMOTION_START) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
        // Not when forced promotion
        if (move.to < WHITE_PAWN_LANCE_FORECED_PROMOTION_START) {
          move.promotion = 0;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
      }
    }
  }

  // Bishop moves
  {
    pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
             notPromoted;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      move.from = iterator.GetCurrentSquare();
      moves =
          (GPU::getDiagRightAttacks(static_cast<Square>(move.from), occupied) |
           GPU::getDiagLeftAttacks(static_cast<Square>(move.from), occupied)) &
          validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
        // Promotion
        if (move.to >= WHITE_PROMOTION_START ||
            move.from >= WHITE_PROMOTION_START) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
      }
    }
  }

  // Rook moves
  {
    pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
             notPromoted;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      move.from = iterator.GetCurrentSquare();
      moves = (GPU::getRankAttacks(static_cast<Square>(move.from), occupied) |
               GPU::getFileAttacks(static_cast<Square>(move.from), occupied)) &
              validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
        // Promotion
        if (move.to >= WHITE_PROMOTION_START ||
            move.from >= WHITE_PROMOTION_START) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
      }
    }
  }
  move.promotion = 0;

  // Horse moves
  {
    pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
             board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      move.from = iterator.GetCurrentSquare();
      Bitboard horse(static_cast<Square>(move.from));
      moves =
          (GPU::getDiagRightAttacks(static_cast<Square>(move.from), occupied) |
           GPU::getDiagLeftAttacks(static_cast<Square>(move.from), occupied) |
           moveN(horse) | moveE(horse) | moveS(horse) | moveW(horse)) &
          validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
  }

  // Dragon moves
  {
    pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
             board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      move.from = iterator.GetCurrentSquare();
      Bitboard dragon(static_cast<Square>(move.from));
      moves =
          (GPU::getRankAttacks(static_cast<Square>(move.from), occupied) |
           GPU::getFileAttacks(static_cast<Square>(move.from), occupied) |
           moveNW(pieces) | moveNE(pieces) | moveSE(pieces) | moveSW(pieces)) &
          validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
  }

  // King moves
  {
    pieces = board[BB::Type::KING] & board[BB::Type::ALL_WHITE];
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      move.from = iterator.GetCurrentSquare();
      moves =
          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
          ~attackedByEnemy & ~board[BB::Type::ALL_WHITE];
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
  }

  // Generate Drop moves
  {
    Bitboard legalDropSpots;
    // Pawns
    if (board.inHand.pieceNumber.WhitePawn > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last rank
      legalDropSpots[BOTTOM] &= ~BOTTOM_RANK;
      // Cannot drop to give checkmate
      pieces = board[BB::Type::KING] & board[BB::Type::ALL_BLACK];
      // All valid enemy king moves
      moves =
          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
          ~ourAttacks & ~board[BB::Type::ALL_BLACK];
      // If there is only one spot pawn cannot block it
      if (d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
              d_popcount(moves[BOTTOM]) ==
          1) {
        legalDropSpots &= ~moveN(moves);
      }
      // Cannot drop on file with other pawn
      Bitboard validFiles;
      for (int fileIdx = 0; fileIdx < BOARD_DIM; fileIdx++) {
        Bitboard file = getFullFile(fileIdx);
        if (!(file & board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
              notPromoted)) {
          validFiles |= file;
        }
      }
      legalDropSpots &= validFiles;
      movesIterator.Init(legalDropSpots);
      move.from = WHITE_PAWN_DROP;
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
    if (board.inHand.pieceNumber.WhiteLance > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last rank
      legalDropSpots[BOTTOM] &= ~BOTTOM_RANK;
      movesIterator.Init(legalDropSpots);
      move.from = WHITE_LANCE_DROP;
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
    if (board.inHand.pieceNumber.WhiteKnight > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last two ranks
      legalDropSpots[BOTTOM] &= TOP_RANK;
      movesIterator.Init(legalDropSpots);
      move.from = WHITE_KNIGHT_DROP;
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
    legalDropSpots = validMoves & ~occupied;
    movesIterator.Init(legalDropSpots);
    while (movesIterator.d_Next()) {
      if (board.inHand.pieceNumber.WhiteSilverGeneral > 0) {
        move.from = WHITE_SILVER_GENERAL_DROP;
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
      if (board.inHand.pieceNumber.WhiteGoldGeneral > 0) {
        move.from = WHITE_GOLD_GENERAL_DROP;
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
      if (board.inHand.pieceNumber.WhiteBishop > 0) {
        move.from = WHITE_BISHOP_DROP;
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
      if (board.inHand.pieceNumber.WhiteRook > 0) {
        move.from = WHITE_ROOK_DROP;
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
  }

  if (moveNumber != numberOfMoves) {
    printf("Invalid number of moves generated. Should be %d, got %d ",
           numberOfMoves, moveNumber);
  }
}

__global__ void generateBlackMovesKernel(uint32_t size,
                                         int16_t movesPerBoard,
                                         Board* startBoard,
                                         Move* inMoves,
                                         uint32_t* inOffsets,
                                         uint32_t* inBitboards,
                                         Move* outMoves) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size)
    return;
  Board board = *startBoard;
  uint32_t movesOffset = inOffsets[index];
  uint32_t numberOfMoves = inOffsets[index + 1] - movesOffset;
  uint32_t allMovesSize = inOffsets[size] - inOffsets[0];
  for (int m = 0; m < movesPerBoard; m++) {
    Move move = inMoves[m * size + index];
    makeMove(board, move);
    for (int i = 0; i < numberOfMoves; i++) {
      outMoves[movesOffset + i] = move;
    }
    movesOffset += allMovesSize;
  }
  Bitboard validMoves, pinned, attackedByEnemy;
  validMoves[TOP] = inBitboards[index];
  validMoves[MID] = inBitboards[size + index];
  validMoves[BOTTOM] = inBitboards[2 * size + index];
  attackedByEnemy[TOP] = inBitboards[3 * size + index];
  attackedByEnemy[MID] = inBitboards[4 * size + index];
  attackedByEnemy[BOTTOM] = inBitboards[5 * size + index];
  pinned[TOP] = inBitboards[6 * size + index];
  pinned[MID] = inBitboards[7 * size + index];
  pinned[BOTTOM] = inBitboards[8 * size + index];

  Bitboard notPromoted = ~board[BB::Type::PROMOTED];
  Bitboard pieces, moves, ourAttacks;
  Bitboard occupied = board[BB::Type::ALL_WHITE] | board[BB::Type::ALL_BLACK];
  BitboardIterator movesIterator, iterator;
  Move move;
  uint32_t moveNumber = 0;
  // Pawn moves
  {
    pieces = ~pinned & board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
             notPromoted;
    moves = moveN(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + S;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
      // Not when forced promotion
      if (move.to > BLACK_PAWN_LANCE_FORCE_PROMOTION_END) {
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
  }
  // Knight moves
  {
    pieces = ~pinned & board[BB::Type::KNIGHT] & board[BB::Type::ALL_BLACK] &
             notPromoted;
    moves = moveN(moveNE(pieces)) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + S + SW;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
      // Not when forced promotion
      if (move.to > BLACK_HORSE_FORCED_PROMOTION_END) {
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
    moves = moveN(moveNW(pieces)) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + S + SE;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
      // Not when forced promotion
      if (move.to > BLACK_HORSE_FORCED_PROMOTION_END) {
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
  }
  // SilverGenerals moves
  {
    pieces = ~pinned & board[BB::Type::SILVER_GENERAL] &
             board[BB::Type::ALL_BLACK] & notPromoted;
    moves = moveN(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + S;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;
      moveNumber++;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
    moves = moveNE(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + SW;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;
      moveNumber++;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
    moves = moveNW(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + SE;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;
      moveNumber++;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
    moves = moveSE(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + NW;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;
      moveNumber++;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END || move.from <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
    moves = moveSW(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + NE;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;
      moveNumber++;
      // Promotion
      if (move.to <= BLACK_PROMOTION_END || move.from <= BLACK_PROMOTION_END) {
        move.promotion = 1;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
  }
  // GoldGenerals moves
  {
    pieces = ~pinned &
             (board[BB::Type::GOLD_GENERAL] |
              ((board[BB::Type::PAWN] | board[BB::Type::LANCE] |
                board[BB::Type::KNIGHT] | board[BB::Type::SILVER_GENERAL]) &
               board[BB::Type::PROMOTED])) &
             board[BB::Type::ALL_BLACK];
    moves = moveN(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + S;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;
      moveNumber++;
    }
    moves = moveNE(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + SW;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;
      moveNumber++;
    }
    moves = moveNW(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + SE;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;
      moveNumber++;
    }
    moves = moveE(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + W;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;
      moveNumber++;
    }
    moves = moveW(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + E;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;
      moveNumber++;
    }
    moves = moveS(pieces) & validMoves;
    ourAttacks |= moves;
    movesIterator.Init(moves);
    while (movesIterator.d_Next()) {
      move.to = movesIterator.GetCurrentSquare();
      move.from = move.to + N;
      move.promotion = 0;
      outMoves[movesOffset + moveNumber] = move;
      moveNumber++;
    }
  }
  // Lances moves
  {
    pieces = ~pinned & board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] &
             notPromoted;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      move.from = iterator.GetCurrentSquare();
      moves = GPU::getFileAttacks(static_cast<Square>(move.from), occupied) &
              GPU::getRankMask(squareToRank(static_cast<Square>(move.from))) &
              validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        // Promotion
        if (move.to <= BLACK_PROMOTION_END) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
        // Not when forced promotion
        if (move.to > BLACK_PAWN_LANCE_FORCE_PROMOTION_END) {
          move.promotion = 0;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
      }
    }
  }
  // Bishop moves
  {
    pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
             notPromoted;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      move.from = iterator.GetCurrentSquare();
      moves =
          (GPU::getDiagRightAttacks(static_cast<Square>(move.from), occupied) |
           GPU::getDiagLeftAttacks(static_cast<Square>(move.from), occupied)) &
          validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
        // Promotion
        if (move.to <= BLACK_PROMOTION_END ||
            move.from <= BLACK_PROMOTION_END) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
      }
    }
  }
  // Rook moves
  {
    pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
             notPromoted;
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      move.from = iterator.GetCurrentSquare();
      moves = (GPU::getRankAttacks(static_cast<Square>(move.from), occupied) |
               GPU::getFileAttacks(static_cast<Square>(move.from), occupied)) &
              validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        move.promotion = 0;
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
        // Promotion
        if (move.to <= BLACK_PROMOTION_END ||
            move.from <= BLACK_PROMOTION_END) {
          move.promotion = 1;
          outMoves[movesOffset + moveNumber] = move;
          moveNumber++;
        }
      }
    }
  }
  move.promotion = 0;
  // Horse moves
  {
    pieces = ~pinned & board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
             board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      move.from = iterator.GetCurrentSquare();
      Bitboard horse(static_cast<Square>(move.from));
      moves =
          (GPU::getDiagRightAttacks(static_cast<Square>(move.from), occupied) |
           GPU::getDiagLeftAttacks(static_cast<Square>(move.from), occupied) |
           moveN(horse) | moveE(horse) | moveS(horse) | moveW(horse)) &
          validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
  }
  // Dragon moves
  {
    pieces = ~pinned & board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
             board[BB::Type::PROMOTED];
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      move.from = iterator.GetCurrentSquare();
      Bitboard dragon(static_cast<Square>(move.from));
      moves =
          (GPU::getRankAttacks(static_cast<Square>(move.from), occupied) |
           GPU::getFileAttacks(static_cast<Square>(move.from), occupied) |
           moveNW(dragon) | moveNE(dragon) | moveSE(dragon) | moveSW(dragon)) &
          validMoves;
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
  }
  // King moves
  {
    pieces = board[BB::Type::KING] & board[BB::Type::ALL_BLACK];
    iterator.Init(pieces);
    while (iterator.d_Next()) {
      move.from = iterator.GetCurrentSquare();
      moves =
          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
          ~attackedByEnemy & ~board[BB::Type::ALL_BLACK];
      ourAttacks |= moves;
      movesIterator.Init(moves);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
  }
  // Drop moves
  {
    Bitboard legalDropSpots;
    // Pawns
    if (board.inHand.pieceNumber.BlackPawn > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last rank
      legalDropSpots[TOP] &= ~TOP_RANK;
      // Cannot drop to give checkmate
      pieces = board[BB::Type::KING] & board[BB::Type::ALL_WHITE];
      // All valid enemy king moves
      moves =
          (moveNW(pieces) | moveN(pieces) | moveNE(pieces) | moveE(pieces) |
           moveSE(pieces) | moveS(pieces) | moveSW(pieces) | moveW(pieces)) &
          ~ourAttacks & ~board[BB::Type::ALL_WHITE];
      // If there is only one spot pawn cannot block it
      if (d_popcount(moves[TOP]) + d_popcount(moves[MID]) +
              d_popcount(moves[BOTTOM]) ==
          1) {
        legalDropSpots &= ~moveS(moves);
      }
      // Cannot drop on file with other pawn
      Bitboard validFiles;
      for (int fileIdx = 0; fileIdx < BOARD_DIM; fileIdx++) {
        Bitboard file = getFullFile(fileIdx);
        if (!(file & board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
              notPromoted)) {
          validFiles |= file;
        }
      }
      legalDropSpots &= validFiles;
      move.from = BLACK_PAWN_DROP;
      movesIterator.Init(legalDropSpots);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
    if (board.inHand.pieceNumber.BlackLance > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last rank
      legalDropSpots[TOP] &= ~TOP_RANK;
      move.from = BLACK_LANCE_DROP;
      movesIterator.Init(legalDropSpots);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
    if (board.inHand.pieceNumber.BlackKnight > 0) {
      legalDropSpots = validMoves & ~occupied;
      // Cannot drop on last two ranks
      legalDropSpots[TOP] &= BOTTOM_RANK;
      move.from = BLACK_KNIGHT_DROP;
      movesIterator.Init(legalDropSpots);
      while (movesIterator.d_Next()) {
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
    legalDropSpots = validMoves & ~occupied;
    movesIterator.Init(legalDropSpots);
    while (movesIterator.d_Next()) {
      if (board.inHand.pieceNumber.BlackSilverGeneral > 0) {
        move.from = BLACK_SILVER_GENERAL_DROP;
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
      if (board.inHand.pieceNumber.BlackGoldGeneral > 0) {
        move.from = BLACK_GOLD_GENERAL_DROP;
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
      if (board.inHand.pieceNumber.BlackBishop > 0) {
        move.from = BLACK_BISHOP_DROP;
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
      if (board.inHand.pieceNumber.BlackRook > 0) {
        move.from = BLACK_ROOK_DROP;
        move.to = movesIterator.GetCurrentSquare();
        outMoves[movesOffset + moveNumber] = move;
        moveNumber++;
      }
    }
  }

  if (moveNumber != numberOfMoves) {
    printf("Invalid number of moves generated. Should be %d, got %d ",
           numberOfMoves, moveNumber);
  }
}

__global__ void evaluateBoardsKernel(uint32_t size,
                                     int16_t movesPerBoard,
                                     Board* startBoard,
                                     Move* inMoves,
                                     int16_t* outValues) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size)
    return;

  Board board = *startBoard;
  for (int m = 0; m < movesPerBoard; m++) {
    makeMoveWhite(board, inMoves[m * size + index]);
  }
  int16_t whitePoints = 0, blackPoints = 0;
  Bitboard pieces;
  // White
  // Pawns
  pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
           ~board[BB::Type::PROMOTED];
  whitePoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PAWN;
  pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
           board[BB::Type::PROMOTED];
  whitePoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_PAWN;
  whitePoints += board.inHand.pieceNumber.WhitePawn * PieceValue::IN_HAND_LANCE;
  // Lances
  pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] &
           ~board[BB::Type::PROMOTED];
  whitePoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::LANCE;
  pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] &
           board[BB::Type::PROMOTED];
  whitePoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_LANCE;
  whitePoints +=
      board.inHand.pieceNumber.WhiteLance * PieceValue::IN_HAND_LANCE;
  // Knights
  pieces = board[BB::Type::KNIGHT] & board[BB::Type::ALL_WHITE] &
           ~board[BB::Type::PROMOTED];
  whitePoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::KNIGHT;
  pieces = board[BB::Type::KNIGHT] & board[BB::Type::ALL_WHITE] &
           board[BB::Type::PROMOTED];
  whitePoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_KNIGHT;
  whitePoints +=
      board.inHand.pieceNumber.WhiteKnight * PieceValue::IN_HAND_KNIGHT;
  // SilverGenerals
  pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_WHITE] &
           ~board[BB::Type::PROMOTED];
  whitePoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::SILVER_GENERAL;
  pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_WHITE] &
           board[BB::Type::PROMOTED];
  whitePoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_SILVER_GENERAL;
  whitePoints += board.inHand.pieceNumber.WhiteSilverGeneral *
                 PieceValue::IN_HAND_SILVER_GENERAL;
  // GoldGenerals
  pieces = board[BB::Type::GOLD_GENERAL] & board[BB::Type::ALL_WHITE] &
           ~board[BB::Type::PROMOTED];
  whitePoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::GOLD_GENERAL;
  whitePoints += board.inHand.pieceNumber.WhiteGoldGeneral *
                 PieceValue::IN_HAND_GOLD_GENERAL;
  // Bishops
  pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
           ~board[BB::Type::PROMOTED];
  whitePoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::BISHOP;
  pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
           board[BB::Type::PROMOTED];
  whitePoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_BISHOP;
  whitePoints +=
      board.inHand.pieceNumber.WhiteBishop * PieceValue::IN_HAND_BISHOP;
  // Rooks
  pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
           ~board[BB::Type::PROMOTED];
  whitePoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::ROOK;
  pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
           board[BB::Type::PROMOTED];
  whitePoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_ROOK;
  whitePoints += board.inHand.pieceNumber.WhiteRook * PieceValue::IN_HAND_ROOK;

  // Black
  // Pawns
  pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
           ~board[BB::Type::PROMOTED];
  blackPoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PAWN;
  pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
           board[BB::Type::PROMOTED];
  blackPoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_PAWN;
  blackPoints += board.inHand.pieceNumber.BlackPawn * PieceValue::IN_HAND_LANCE;
  // Lances
  pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] &
           ~board[BB::Type::PROMOTED];
  blackPoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::LANCE;
  pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] &
           board[BB::Type::PROMOTED];
  blackPoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_LANCE;
  blackPoints +=
      board.inHand.pieceNumber.BlackLance * PieceValue::IN_HAND_LANCE;
  // Knights
  pieces = board[BB::Type::KNIGHT] & board[BB::Type::ALL_BLACK] &
           ~board[BB::Type::PROMOTED];
  blackPoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::KNIGHT;
  pieces = board[BB::Type::KNIGHT] & board[BB::Type::ALL_BLACK] &
           board[BB::Type::PROMOTED];
  blackPoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_KNIGHT;
  blackPoints +=
      board.inHand.pieceNumber.BlackKnight * PieceValue::IN_HAND_KNIGHT;
  // SilverGenerals
  pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_BLACK] &
           ~board[BB::Type::PROMOTED];
  blackPoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::SILVER_GENERAL;
  pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_BLACK] &
           board[BB::Type::PROMOTED];
  blackPoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_SILVER_GENERAL;
  blackPoints += board.inHand.pieceNumber.BlackSilverGeneral *
                 PieceValue::IN_HAND_SILVER_GENERAL;
  // GoldGenerals
  pieces = board[BB::Type::GOLD_GENERAL] & board[BB::Type::ALL_BLACK] &
           ~board[BB::Type::PROMOTED];
  blackPoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::GOLD_GENERAL;
  blackPoints += board.inHand.pieceNumber.BlackGoldGeneral *
                 PieceValue::IN_HAND_GOLD_GENERAL;
  // Bishops
  pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
           ~board[BB::Type::PROMOTED];
  blackPoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::BISHOP;
  pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
           board[BB::Type::PROMOTED];
  blackPoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_BISHOP;
  blackPoints +=
      board.inHand.pieceNumber.BlackBishop * PieceValue::IN_HAND_BISHOP;
  // Rooks
  pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
           ~board[BB::Type::PROMOTED];
  blackPoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::ROOK;
  pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
           board[BB::Type::PROMOTED];
  blackPoints += (d_popcount(pieces[TOP]) + d_popcount(pieces[MID]) +
                  d_popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_ROOK;
  blackPoints += board.inHand.pieceNumber.BlackRook * PieceValue::IN_HAND_ROOK;

  outValues[index] = whitePoints - blackPoints;
}

__global__ void gatherValuesMaxKernel(uint32_t size,
                                      uint16_t depth,
                                      uint32_t* inOffsets,
                                      int16_t* inValues,
                                      int16_t* outValues,
                                      uint32_t* bestIndex) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size)
    return;
  uint32_t childrenStart = inOffsets[index];
  uint32_t childrenSize = inOffsets[index + 1] - childrenStart;
  if (childrenSize > 500)
    printf("Index %d Children start: %d, childrenSize: %d ", index,
           childrenStart, childrenSize);
  int16_t maxValue = INT16_MIN;
  for (int i = 0; i < childrenSize; i++) {
    int16_t value = inValues[childrenStart + i];
    if (value > maxValue) {
      maxValue = value;
      if (index == 0) {
        *bestIndex = i;
      }
    }
  }
  if (childrenSize == 0) {
    maxValue = -PieceValue::MATE + depth;
  }
  outValues[index] = maxValue;
}

__global__ void gatherValuesMinKernel(uint32_t size,
                                      uint16_t depth,
                                      uint32_t* inOffsets,
                                      int16_t* inValues,
                                      int16_t* outValues,
                                      uint32_t* bestIndex) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size)
    return;
  uint32_t childrenStart = inOffsets[index];
  uint32_t childrenSize = inOffsets[index + 1] - childrenStart;
  if (childrenSize > 500)
    printf("Index %d Children start: %d, childrenSize: %d ", index,
           childrenStart, childrenSize);
  int16_t minValue = INT16_MAX;
  for (int i = 0; i < childrenSize; i++) {
    int16_t value = inValues[childrenStart + i];
    if (value < minValue) {
      minValue = value;
      if (index == 0) {
        *bestIndex = i;
      }
    }
  }
  if (childrenSize == 0) {
    minValue = PieceValue::MATE - depth;
  }
  outValues[index] = minValue;
}

int GPU::countWhiteMoves(uint32_t size,
                         int16_t movesPerBoard,
                         Board* startBoard,
                         Move* inMoves,
                         uint32_t* outOffsets,
                         uint32_t* outBitboards) {
  int numberOfBlocks = calculateNumberOfBlocks(size);
  countWhiteMovesKernel<<<numberOfBlocks, THREADS_COUNT>>>(
      size, movesPerBoard, startBoard, inMoves, outOffsets, outBitboards);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return 0;
}

int GPU::countBlackMoves(uint32_t size,
                         int16_t movesPerBoard,
                         Board* startBoard,
                         Move* inMoves,
                         uint32_t* outOffsets,
                         uint32_t* outBitboards) {
  int numberOfBlocks = calculateNumberOfBlocks(size);
  countBlackMovesKernel<<<numberOfBlocks, THREADS_COUNT>>>(
      size, movesPerBoard, startBoard, inMoves, outOffsets, outBitboards);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return 0;
}

int GPU::generateWhiteMoves(uint32_t size,
                            int16_t movesPerBoard,
                            Board* startBoard,
                            Move* inMoves,
                            uint32_t* inOffsets,
                            uint32_t* inBitboards,
                            Move* outMoves) {
  int numberOfBlocks = calculateNumberOfBlocks(size);
  generateWhiteMovesKernel<<<numberOfBlocks, THREADS_COUNT>>>(
      size, movesPerBoard, startBoard, inMoves, inOffsets, inBitboards,
      outMoves);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return 0;
}

int GPU::generateBlackMoves(uint32_t size,
                            int16_t movesPerBoard,
                            Board* startBoard,
                            Move* inMoves,
                            uint32_t* inOffsets,
                            uint32_t* inBitboards,
                            Move* outMoves) {
  int numberOfBlocks = calculateNumberOfBlocks(size);
  generateBlackMovesKernel<<<numberOfBlocks, THREADS_COUNT>>>(
      size, movesPerBoard, startBoard, inMoves, inOffsets, inBitboards,
      outMoves);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return 0;
}

int GPU::evaluateBoards(uint32_t size,
                        int16_t movesPerBoard,
                        Board* startBoard,
                        Move* inMoves,
                        int16_t* outValues) {
  int numberOfBlocks = calculateNumberOfBlocks(size);
  evaluateBoardsKernel<<<numberOfBlocks, THREADS_COUNT>>>(
      size, movesPerBoard, startBoard, inMoves, outValues);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return 0;
}

int GPU::gatherValuesMax(uint32_t size,
                         uint16_t depth,
                         uint32_t* inOffsets,
                         int16_t* inValues,
                         int16_t* outValues,
                         uint32_t* bestIndex) {
  int numberOfBlocks = calculateNumberOfBlocks(size);
  gatherValuesMaxKernel<<<numberOfBlocks, THREADS_COUNT>>>(
      size, depth, inOffsets, inValues, outValues, bestIndex);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return 0;
}

int GPU::gatherValuesMin(uint32_t size,
                         uint16_t depth,
                         uint32_t* inOffsets,
                         int16_t* inValues,
                         int16_t* outValues,
                         uint32_t* bestIndex) {
  int numberOfBlocks = calculateNumberOfBlocks(size);
  gatherValuesMinKernel<<<numberOfBlocks, THREADS_COUNT>>>(
      size, depth, inOffsets, inValues, outValues, bestIndex);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return 0;
}
}  // namespace engine
}  // namespace shogi