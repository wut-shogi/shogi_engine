#include "Board.h"

Board Boards::startingBoard = {{
    Bitboards::STARTING_PAWN,
    Bitboards::STARTING_LANCE,
    Bitboards::STARTING_KNIGHT,
    Bitboards::STARTING_SILVER_GENERAL,
    Bitboards::STARTING_GOLD_GENERAL,
    Bitboards::STARTING_BISHOP,
    Bitboards::STARTING_ROOK,
    Bitboards::STARTING_KING,
    Bitboards::STARTING_PROMOTED,
    Bitboards::STARTING_ALL_WHITE,
    Bitboards::STARTING_ALL_BLACK,
}};