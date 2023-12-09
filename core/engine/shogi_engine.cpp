#include "shogi_engine.h"
#include "MoveGen.h"

BSTR getAllLegalMoves(const char* SFENstring) {
  bool isWhite;
  shogi::engine::Board board =
      shogi::engine::Board::FromSFEN(SFENstring, isWhite);
  std::vector<std::string> moves =
      shogi::engine::getAllLegalMovesUSI(board, isWhite);
  std::string movesString = "";
  for (std::string move : moves) {
    movesString += move + '|';
  }
  movesString.pop_back();
  std::wstring movesWstring(movesString.begin(), movesString.end());
  return SysAllocString(movesWstring.c_str());
}

BSTR getBestMove(const char* SFENstring,
                 unsigned int maxDepth,
                 unsigned int maxTime) {
  bool isWhite;
  shogi::engine::Board board =
      shogi::engine::Board::FromSFEN(SFENstring, isWhite);
  std::string bestMove =
      shogi::engine::getBestMoveUSI(board, isWhite, maxDepth, maxTime);
  std::wstring bestMoveWstring(bestMove.begin(), bestMove.end());
  return SysAllocString(bestMoveWstring.c_str());
}
