#pragma once
#include "Board.h"
#include "Rules.h"

Bitboard moveN(Bitboard bb);
Bitboard moveNE(Bitboard bb);
Bitboard moveE(Bitboard bb);
Bitboard moveSE(Bitboard bb);
Bitboard moveS(Bitboard bb);
Bitboard moveSW(Bitboard bb);
Bitboard moveW(Bitboard bb);
Bitboard moveNW(Bitboard bb);

/// Atacks bitboards
void whitePawnsAttackBitboards(const Bitboard pawns, Bitboard* outAttacksBitboards);
void blackPawnsAttackBitboards(const Bitboard pawns,
                               Bitboard* outAttacksBitboards);
void whiteKnightsAttackBitboards(const Bitboard knights,
                                 Bitboard* outAttacksBitboards);
void blackKnightsAttackBitboards(const Bitboard knights,
                                 Bitboard* outAttacksBitboards);
void whiteSilverGeneralsAttackBitboards(const Bitboard silverGenerals,
                                        Bitboard* outAttacksBitboards);
void blackSilverGeneralsAttackBitboards(const Bitboard silverGenerals,
                                        Bitboard* outAttacksBitboards);
void whiteGoldGeneralsAttackBitboards(const Bitboard goldGenerals,
                                      Bitboard* outAttacksBitboards);
void blackGoldGeneralsAttackBitboards(const Bitboard goldGenerals,
                                      Bitboard* outAttacksBitboards);
void kingAttackBitboards(const Square king, Bitboard* outAttacksBitboards);
void whiteLanceAttackBitboards(const Square lance,
                               const Bitboard& occupiedRot90,
                               Bitboard* outAttacksBitboards);
void blackLanceAttackBitboards(const Square lance,
                                   const Bitboard& occupiedRot90,
                                   Bitboard* outAttacksBitboards);
void bishopAttackBitboards(const Square bishop,
                                  const Bitboard& occupiedRot45Right,
                               const Bitboard& occupiedRot45Left,
                               Bitboard* outAttacksBitboards);
void rookAttackBitboards(const Square Rook,
                                const Bitboard& occupied,
                             const Bitboard& occupiedRot90,
                             Bitboard* outAttacksBitboards);
void horseAttackBitboards(const Square horse,
                            const Bitboard& occupiedRot45Right,
                              const Bitboard& occupiedRot45Left,
                              Bitboard* outAttacksBitboards);
void dragonAttackBitboards(const Square dragon,
                             const Bitboard& occupied,
                               const Bitboard& occupiedRot90,
                               Bitboard* outAttacksBitboards);
//// Move count
size_t countWhitePawnsMoves(const Bitboard pawns, const Bitboard& validMoves);
size_t countBlackPawnsMoves(const Bitboard pawns, const Bitboard& validMoves);
size_t countWhiteKnightsMoves(const Bitboard knights,
                             const Bitboard& validMoves);
size_t countBlackKnightsMoves(const Bitboard knights,
                             const Bitboard& validMoves);
size_t countWhiteSilverGeneralsMoves(const Bitboard silverGenerals,
                                    const Bitboard& validMoves);
size_t countBlackSilverGeneralsMoves(const Bitboard silverGenerals,
                                    const Bitboard& validMoves);
size_t countWhiteGoldGeneralsMoves(const Bitboard goldGenerals,
                                  const Bitboard& validMoves);
size_t countBlackGoldGeneralsMoves(const Bitboard goldGenerals,
                                  const Bitboard& validMoves);
size_t countKingMoves(const Square king, const Bitboard& validMoves);
size_t countWhiteLancesMoves(const Square lance1, const Square lance2,
                            const Bitboard& validMoves,
                            const Bitboard& occupiedRot90);
size_t countBlackLancesMoves(const Square lance1,
                            const Square lance2,
                            const Bitboard& validMoves,
                            const Bitboard& occupiedRot90);
size_t countWhiteBishopMoves(const Square bishop,
                             const Bitboard& validMoves,
                             const Bitboard& occupiedRot45Right,
                             const Bitboard& occupiedRot45Left);
size_t countBlackBishopMoves(const Square bishop,
                             const Bitboard& validMoves,
                             const Bitboard& occupiedRot45Right,
                             const Bitboard& occupiedRot45Left);
size_t countWhiteRookMoves(const Square rook,
                           const Bitboard& validMoves,
                           const Bitboard& occupied,
                           const Bitboard& occupiedRot90);
size_t countBlackRookMoves(const Square rook,
                           const Bitboard& validMoves,
                           const Bitboard& occupied,
                           const Bitboard& occupiedRot90);
size_t countHorseMoves(const Square horse,
                       const Bitboard& validMoves,
                       const Bitboard& occupiedRot45Right,
                       const Bitboard& occupiedRot45Left);
size_t countDragonMoves(const Square dragon,
                        const Bitboard& validMoves,
                        const Bitboard& occupied,
                        const Bitboard& occupiedRot90);

size_t countDropMoves(const PlayerInHandPieces& inHand,
                      const Bitboard& freeSquares,
                      const Bitboard ownPawns,
                      const Bitboard enemyKing,
                      bool isWhite);