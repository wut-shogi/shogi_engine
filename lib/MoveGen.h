#pragma once

#include "Bitboard.h"
#include "Rules.h"

// Non-sliding pieces
static std::array<Bitboard, BOARD_SIZE - BOARD_DIM> initWhitePawnAttacks();
static std::array<Bitboard, BOARD_SIZE - 2 * BOARD_DIM>
initWhiteKnightAttacks();
static std::array<Bitboard, BOARD_SIZE> initWhiteSilverGeneralAttacks();
static std::array<Bitboard, BOARD_SIZE> initWhiteGoldGeneralAttacks();
static std::array<Bitboard, BOARD_SIZE - BOARD_DIM> initBlackPawnAttacks();
static std::array<Bitboard, BOARD_SIZE - 2 * BOARD_DIM>
initBlackKnightAttacks();
static std::array<Bitboard, BOARD_SIZE> initBlackSilverGeneralAttacks();
static std::array<Bitboard, BOARD_SIZE> initBlackGoldGeneralAttacks();
static std::array<Bitboard, BOARD_SIZE> initkKingAttacks();

static std::array<Bitboard, BOARD_SIZE> WhitePawnAttacks;
static std::array<Bitboard, BOARD_SIZE> WhiteKnightAttacks;
static std::array<Bitboard, BOARD_SIZE> WhiteSilverGeneralAttacks;
static std::array<Bitboard, BOARD_SIZE> WhiteGoldGeneralAttacks;
static std::array<Bitboard, BOARD_SIZE> BlackPawnAttacks;
static std::array<Bitboard, BOARD_SIZE> BlackKnightAttacks;
static std::array<Bitboard, BOARD_SIZE> BlackSilverGeneralAttacks;
static std::array<Bitboard, BOARD_SIZE> BlackGoldGeneralAttacks;
static std::array<Bitboard, BOARD_SIZE> KingAttacks;

// Sliding pieces
static std::array<std::array<Bitboard, BOARD_SIZE>, 128> initRankAttacks();
static std::array<std::array<Bitboard, BOARD_SIZE>, 128> initFileAttacks();
static std::array<std::array<Bitboard, BOARD_SIZE>, 128> initDiagRightAttacks();
static std::array<std::array<Bitboard, BOARD_SIZE>, 128> initDiagLeftAttacks();

static std::array<std::array<Bitboard, BOARD_SIZE>, 128> RankAttacks;
static std::array<std::array<Bitboard, BOARD_SIZE>, 128> FileAttacks;
static std::array<std::array<Bitboard, BOARD_SIZE>, 128> DiagRightAttacks;
static std::array<std::array<Bitboard, BOARD_SIZE>, 128> DiagLeftAttacks;