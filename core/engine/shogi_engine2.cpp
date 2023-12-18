#include "shogi_engine2.h"
#include "MoveGen.h"

void getAllLegalMoves(const char* input, char* output) {
  bool isWhite;
  shogi::engine::Board board = shogi::engine::Board::FromSFEN(input, isWhite);
  std::vector<std::string> moves =
      shogi::engine::getAllLegalMovesUSI(board, isWhite);
  std::string movesString = "";
  for (std::string move : moves) {
    movesString += move + '|';
  }
  movesString.pop_back();
  std::wstring movesWstring(movesString.begin(), movesString.end());
  strcpy(output, movesString.c_str());
  //return SysAllocString(movesWstring.c_str());
}

void getBestMove(const char* input,
                 unsigned int maxDepth,
                 unsigned int maxTime,
                 char* output) {
  bool isWhite;
  shogi::engine::Board board = shogi::engine::Board::FromSFEN(input, isWhite);
  std::string bestMove =
      shogi::engine::getBestMoveUSI(board, isWhite, maxDepth, maxTime);
  strcpy(output, bestMove.c_str());
}
