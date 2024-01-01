#include "../../include/shogi_engine.h"
#include "search.h"

static bool afterInit = false;

static std::string getBestMoveUSI(const shogi::engine::Board& board,
                                  bool isWhite,
                                  unsigned int maxDepth,
                                  unsigned int maxTime = 0,
                                  bool useGPU = true) {
  shogi::engine::Move bestMove = shogi::engine::SEARCH::GetBestMove(
      board, isWhite, maxDepth, maxTime,
      useGPU ? shogi::engine::SEARCH::GPU : shogi::engine::SEARCH::CPU);
  return shogi::engine::moveToUSI(bestMove);
}

bool init() {
  if (!afterInit)
    afterInit = shogi::engine::SEARCH::init();
  return afterInit;
}

void cleanup() {
  if (afterInit)
    shogi::engine::SEARCH::cleanup();
  afterInit = false;
}

int getAllLegalMoves(const char* SFENstring, char* output) {
  bool isWhite;
  shogi::engine::Board board =
      shogi::engine::Board::FromSFEN(SFENstring, isWhite);
  std::string movesString = "";
  if (afterInit) {
    shogi::engine::CPU::MoveList moves(board, isWhite);
    for (const auto& move : moves) {
      movesString += shogi::engine::moveToUSI(move) + '|';
    }
    if (!movesString.empty())
      movesString.pop_back();

    strcpy(output, movesString.data());
  }
  return movesString.size();
}

int getBestMove(const char* SFENstring,
                unsigned int maxDepth,
                unsigned int maxTime,
                bool useGPU,
                char* output) {
  bool isWhite;
  shogi::engine::Board board =
      shogi::engine::Board::FromSFEN(SFENstring, isWhite);
  std::string bestMoveString = "";
  if (afterInit) {
    bestMoveString = getBestMoveUSI(board, isWhite, maxDepth, maxTime, useGPU);

    strcpy(output, bestMoveString.data());
  }
  return bestMoveString.size();
}
