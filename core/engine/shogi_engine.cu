#include "../../include/shogi_engine.h"
#include "search.h"

static bool afterInit = false;

static std::string getBestMoveUSI(const shogi::engine::Board& board,
                                  bool isWhite,
                                  unsigned int maxDepth,
                                  unsigned int maxTime = 0) {
  shogi::engine::Move bestMove =
      shogi::engine::search::GetBestMove(board, isWhite, 0, maxDepth);
  return shogi::engine::moveToUSI(bestMove);
}

bool init() {
  if (!afterInit)
    afterInit = shogi::engine::search::init();
  return afterInit;
}

void cleanup() {
  if (afterInit)
    shogi::engine::search::cleanup();
  afterInit = false;
}

BSTR getAllLegalMoves(const char* SFENstring) {
  bool isWhite;
  shogi::engine::Board board =
      shogi::engine::Board::FromSFEN(SFENstring, isWhite);
  std::string movesString = "";
  if (afterInit) {
    shogi::engine::CPU::MoveList moves(board, isWhite);
    for (const auto& move : moves) {
      movesString += shogi::engine::moveToUSI(move) + '|';
    }
    movesString.pop_back();
  }
  std::wstring movesWstring(movesString.begin(), movesString.end());
  return SysAllocString(movesWstring.c_str());
}

BSTR getBestMove(const char* SFENstring,
                 unsigned int maxDepth,
                 unsigned int maxTime) {
  bool isWhite;
  shogi::engine::Board board =
      shogi::engine::Board::FromSFEN(SFENstring, isWhite);
  std::string bestMoveString = "";
  if (afterInit)
    bestMoveString = getBestMoveUSI(board, isWhite, maxDepth);
  std::wstring bestMoveWstring(bestMoveString.begin(), bestMoveString.end());
  return SysAllocString(bestMoveWstring.c_str());
}
