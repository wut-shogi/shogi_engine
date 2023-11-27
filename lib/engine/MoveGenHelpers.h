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
//// Move count
size_t countWhitePawnMoves(const Bitboard pawns, const Bitboard& validMoves);
size_t countBlackPawnMoves(const Bitboard pawns, const Bitboard& validMoves);
size_t countWhiteKnightMoves(const Bitboard knights,
                             const Bitboard& validMoves);
size_t countBlackKnightMoves(const Bitboard knights,
                             const Bitboard& validMoves);
size_t countWhiteSilverGeneralMoves(const Bitboard silverGenerals,
                                    const Bitboard& validMoves);
size_t countBlackSilverGeneralMoves(const Bitboard silverGenerals,
                                    const Bitboard& validMoves);
size_t countWhiteGoldGeneralMoves(const Bitboard goldGenerals,
                                  const Bitboard& validMoves);
size_t countBlackGoldGeneralMoves(const Bitboard goldGenerals,
                                  const Bitboard& validMoves);
size_t countKingMoves(const Bitboard king, const Bitboard& validMoves);
size_t countWhiteLanceMoves(const Bitboard lances,
                            const Bitboard& validMoves,
                            const Bitboard& occupiedRot90);
size_t countBlackLanceMoves(const Bitboard lances,
                            const Bitboard& validMoves,
                            const Bitboard& occupiedRot90);
size_t countWhiteBishopMoves(const Bitboard bishops,
                             const Bitboard& validMoves,
                             const Bitboard& occupiedRot45Right,
                             const Bitboard& occupiedRot45Left);
size_t countBlackBishopMoves(const Bitboard bishops,
                             const Bitboard& validMoves,
                             const Bitboard& occupiedRot45Right,
                             const Bitboard& occupiedRot45Left);
size_t countWhiteRookMoves(const Bitboard rooks,
                           const Bitboard& validMoves,
                           const Bitboard& occupied,
                           const Bitboard& occupiedRot90);
size_t countBlackRookMoves(const Bitboard rooks,
                           const Bitboard& validMoves,
                           const Bitboard& occupied,
                           const Bitboard& occupiedRot90);
size_t countHorseMoves(const Bitboard horse,
                       const Bitboard& validMoves,
                       const Bitboard& occupiedRot45Right,
                       const Bitboard& occupiedRot45Left);
size_t countDragonMoves(const Bitboard dragon,
                        const Bitboard& validMoves,
                        const Bitboard& occupied,
                        const Bitboard& occupiedRot90);

size_t countDropMoves(const NumberOfPieces& inHand,
                      const Bitboard& freeSquares,
                      const Bitboard ownPawns,
                      const Bitboard enemyKing,
                      bool isWhite);