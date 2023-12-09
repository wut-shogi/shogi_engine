#pragma once
#include "Board.h"
#include "Rules.h"

namespace shogi {
namespace engine {
Bitboard moveN(Bitboard bb);
Bitboard moveNE(Bitboard bb);
Bitboard moveE(Bitboard bb);
Bitboard moveSE(Bitboard bb);
Bitboard moveS(Bitboard bb);
Bitboard moveSW(Bitboard bb);
Bitboard moveW(Bitboard bb);
Bitboard moveNW(Bitboard bb);

const Bitboard& getRankAttacks(const Square& square, const Bitboard& occupied);
const Bitboard& getFileAttacks(const Square& square, const Bitboard& occupied);
const Bitboard& getDiagRightAttacks(const Square& square,
                                    const Bitboard& occupied);
const Bitboard& getDiagLeftAttacks(const Square& square,
                                   const Bitboard& occupied);
const Bitboard& getRankMask(const uint32_t& rank);
const Bitboard& getFileMask(const uint32_t& file);

Bitboard getFullFile(int fileIdx);
}  // namespace engine
}  // namespace shogi