#pragma once
#include <vector>
#include "Board.h"
#include "MoveGenHelpers.h"

struct Move {
  uint16_t from : 7;
  uint16_t to : 7;
  uint16_t promotion : 1;
};

// Returns number of all white moves from position. outValidMoves and
// attackedByEnemy need to be stored for move generation
size_t countWhiteMoves(const Board& board,
                       Bitboard& outValidMoves,
                       Bitboard& attackedByEnemy);

// Returns number of all black moves from position. outValidMoves and
// attackedByEnemy need to be stored for move generation
size_t countBlackMoves(const Board& board,
                       Bitboard& outValidMoves,
                       Bitboard& attackedByEnemy);

// Generates all moves from position. validMoves and attackedByEnemy should be
// taken from previous countWhiteMoves call. movesArray+offset have to have
// enough memmory allocated
void generateWhiteMoves(const Board& board,
                        const Bitboard& validMoves,
                        const Bitboard& attackedByEnemy,
                        Move* movesArray,
                        size_t offset);

// Generates all moves from position. validMoves and attackedByEnemy should be
// taken from previous countBlackMoves call. movesArray+offset have to have
// enough memmory allocated
void generateBlackMoves(const Board& board,
                        const Bitboard& validMoves,
                        const Bitboard& attackedByEnemy,
                        Move* movesArray,
                        size_t offset);

// Returns all legal moves from given position for given player
// Return value is and array of vectors of moves. Each entry in the array stores
// are legal moves from square entryIndex as target square index and boolean
// information if promotion can be done
std::array<std::vector<std::pair<int, bool>>, BOARD_SIZE + 14> getAllLegalMoves(
    const Board& board,
    bool isWhite);
