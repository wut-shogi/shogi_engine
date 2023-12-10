#pragma once
#include <string>
#include <vector>
#include "MoveGenHelpers.h"
namespace shogi {
namespace engine {
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

void makeMove(Board& board, const Move& move);

void generateNextBoards(const Board& board,
                        Move* movesArray,
                        size_t length,
                        Board* newBoardsArray);

std::vector<Move> getAllLegalMoves(const Board& board, bool isWhite);
std::vector<std::string> getAllLegalMovesUSI(const Board& board, bool isWhite);

Move getBestMove(const Board& board,
                 bool isWhite,
                 unsigned int maxDepth,
                 unsigned int maxTime = 0);

std::string getBestMoveUSI(const Board& board,
                           bool isWhite,
                           unsigned int maxDepth,
                           unsigned int maxTime = 0);

std::string moveToString(const Move& move);

}  // namespace engine
}  // namespace shogi
