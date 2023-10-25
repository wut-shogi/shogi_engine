#pragma once

#include "Bitboard.h"
#include "Rules.h"

// Non-sliding pieces
std::array<Bitboard, BOARD_SIZE - BOARD_DIM> initWhitePawnAttacks();
std::array<Bitboard, BOARD_SIZE - 2 * BOARD_DIM>
initWhiteKnightAttacks();
std::array<Bitboard, BOARD_SIZE> initWhiteSilverGeneralAttacks();
std::array<Bitboard, BOARD_SIZE> initWhiteGoldGeneralAttacks();
std::array<Bitboard, BOARD_SIZE - BOARD_DIM> initBlackPawnAttacks();
std::array<Bitboard, BOARD_SIZE - 2 * BOARD_DIM>
initBlackKnightAttacks();
std::array<Bitboard, BOARD_SIZE> initBlackSilverGeneralAttacks();
std::array<Bitboard, BOARD_SIZE> initBlackGoldGeneralAttacks();
std::array<Bitboard, BOARD_SIZE> initKingAttacks();

extern std::array<Bitboard, BOARD_SIZE - BOARD_DIM> WhitePawnAttacks;
extern std::array<Bitboard, BOARD_SIZE - 2 * BOARD_DIM> WhiteKnightAttacks;
extern std::array<Bitboard, BOARD_SIZE> WhiteSilverGeneralAttacks;
extern std::array<Bitboard, BOARD_SIZE> WhiteGoldGeneralAttacks;
extern std::array<Bitboard, BOARD_SIZE - BOARD_DIM> BlackPawnAttacks;
extern std::array<Bitboard, BOARD_SIZE - 2 * BOARD_DIM> BlackKnightAttacks;
extern std::array<Bitboard, BOARD_SIZE> BlackSilverGeneralAttacks;
extern std::array<Bitboard, BOARD_SIZE> BlackGoldGeneralAttacks;
extern std::array<Bitboard, BOARD_SIZE> KingAttacks;

// Sliding pieces
std::array<std::array<Bitboard, BOARD_SIZE>, 128> initRankAttacks();
std::array<std::array<Bitboard, BOARD_SIZE>, 128> initFileAttacks();
std::array<std::array<Bitboard, BOARD_SIZE>, 128> initDiagRightAttacks();
std::array<std::array<Bitboard, BOARD_SIZE>, 128> initDiagLeftAttacks();

extern std::array<std::array<Bitboard, BOARD_SIZE>, 128> RankAttacks;
extern std::array<std::array<Bitboard, BOARD_SIZE>, 128> FileAttacks;
extern std::array<std::array<Bitboard, BOARD_SIZE>, 128> DiagRightAttacks;
extern std::array<std::array<Bitboard, BOARD_SIZE>, 128> DiagLeftAttacks;


int getRankBlockPattern(Bitboard& bb, int fieldIdx);
int getFileBlockPattern(Bitboard& bb, int fieldIdx);
int getDiagRightBlockPattern(Bitboard& bbRot90, int fieldIdx);
int getDiagLeftBlockPattern(Bitboard& bbRot90, int fieldIdx);