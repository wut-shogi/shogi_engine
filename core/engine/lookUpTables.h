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

RUNTYPE uint32_t getRankBlockPattern(const Bitboard& bb,
                                                 Square square);

RUNTYPE uint32_t getFileBlockPattern(const Bitboard& occupied,
                                                 Square square);

RUNTYPE uint32_t getDiagRightBlockPattern(const Bitboard& occupied,
                                                      Square square);

RUNTYPE uint32_t getDiagLeftBlockPattern(const Bitboard& occupied,
                                                     Square square);

RUNTYPE const Bitboard& getRankAttacks(const Square& square,
                                                   const Bitboard& occupied);
RUNTYPE const Bitboard& getFileAttacks(const Square& square,
                                                   const Bitboard& occupied);
RUNTYPE const Bitboard& getDiagRightAttacks(
    const Square& square,
    const Bitboard& occupied);
RUNTYPE const Bitboard& getDiagLeftAttacks(
    const Square& square,
    const Bitboard& occupied);
RUNTYPE const Bitboard& getRankMask(const uint32_t& rank);
RUNTYPE const Bitboard& getFileMask(const uint32_t& file);
}  // namespace LookUpTables
}  // namespace engine
}  // namespace shogi