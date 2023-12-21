#pragma once
#include "Bitboard.h"
namespace shogi {
namespace engine {
namespace LookUpTables {
namespace CPU {
void init();
void cleanup();
}  // namespace CPU
namespace GPU {
int init();
void cleanup();
}  // namespace GPU

__host__ __device__ uint32_t getRankBlockPattern(const Bitboard& bb,
                                                 Square square);

__host__ __device__ uint32_t getFileBlockPattern(const Bitboard& occupied,
                                                 Square square);

__host__ __device__ uint32_t getDiagRightBlockPattern(const Bitboard& occupied,
                                                      Square square);

__host__ __device__ uint32_t getDiagLeftBlockPattern(const Bitboard& occupied,
                                                     Square square);

__host__ __device__ const Bitboard& getRankAttacks(const Square& square,
                                                   const Bitboard& occupied);
__host__ __device__ const Bitboard& getFileAttacks(const Square& square,
                                                   const Bitboard& occupied);
__host__ __device__ const Bitboard& getDiagRightAttacks(
    const Square& square,
    const Bitboard& occupied);
__host__ __device__ const Bitboard& getDiagLeftAttacks(
    const Square& square,
    const Bitboard& occupied);
__host__ __device__ const Bitboard& getRankMask(const uint32_t& rank);
__host__ __device__ const Bitboard& getFileMask(const uint32_t& file);
}  // namespace LookUpTables
}  // namespace engine
}  // namespace shogi