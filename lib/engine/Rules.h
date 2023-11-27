#pragma once
#define BOARD_DIM 9
#define BOARD_SIZE 81
#define NOT_RIGHT_FILE 133955070
#define NOT_LEFT_FILE 66977535
#define LAST_SQUARE 1
#define FIRST_SQUARE 67108864

#define TOP_RANK 133955584
#define MID_RANK 261632
#define BOTTOM_RANK 511

namespace Player {
enum Type { NONE, WHITE, BLACK };
}

namespace Piece {
enum Type {
  PAWN = 0,
  KNIGHT,
  SILVER_GENERAL,
  GOLD_GENERAL,
  KING,
  LANCE,
  BISHOP,
  ROOK,
  PROMOTED_PAWN,
  PROMOTED_KNIGHT,
  PROMOTED_SILVER_GENERAL,
  PROMOTED_LANCE,
  HORSE,
  DRAGON
};
}