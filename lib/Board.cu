#include "Board.h"

Board Boards::STARTING_BOARD() {
  static Board b = {{
      Bitboards::STARTING_PAWN(),
      Bitboards::STARTING_LANCE(),
      Bitboards::STARTING_KNIGHT(),
      Bitboards::STARTING_SILVER_GENERAL(),
      Bitboards::STARTING_GOLD_GENERAL(),
      Bitboards::STARTING_BISHOP(),
      Bitboards::STARTING_ROOK(),
      Bitboards::STARTING_KING(),
      Bitboards::STARTING_PROMOTED(),
      Bitboards::STARTING_ALL_WHITE(),
      Bitboards::STARTING_ALL_BLACK(),
      Rotate90Clockwise(Bitboards::STARTING_ALL_WHITE() |
               Bitboards::STARTING_ALL_BLACK()),
      Rotate45Clockwise(Bitboards::STARTING_ALL_WHITE() |
                        Bitboards::STARTING_ALL_BLACK()),
      Rotate45AntiClockwise(Bitboards::STARTING_ALL_WHITE() |
                        Bitboards::STARTING_ALL_BLACK()),
  }};
  return b;
}