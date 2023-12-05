#include "Board.h"

Board Boards::STARTING_BOARD() {
  InHandPieces inHandPieces;
  inHandPieces.value = 0;
  static Board b = {{
                        Bitboards::STARTING_PAWN(),
                        Bitboards::STARTING_KNIGHT(),
                        Bitboards::STARTING_SILVER_GENERAL(),
                        Bitboards::STARTING_GOLD_GENERAL(),
                        Bitboards::STARTING_KING(),
                        Bitboards::STARTING_LANCE(),
                        Bitboards::STARTING_BISHOP(),
                        Bitboards::STARTING_ROOK(),
                        Bitboards::STARTING_PROMOTED(),
                        Bitboards::STARTING_ALL_WHITE(),
                        Bitboards::STARTING_ALL_BLACK(),
                        Rotate90Clockwise(Bitboards::STARTING_ALL_WHITE() |
                                          Bitboards::STARTING_ALL_BLACK()),
                        Rotate45Clockwise(Bitboards::STARTING_ALL_WHITE() |
                                          Bitboards::STARTING_ALL_BLACK()),
                        Rotate45AntiClockwise(Bitboards::STARTING_ALL_WHITE() |
                                              Bitboards::STARTING_ALL_BLACK()),
                    },
                    inHandPieces};
  return b;
}
