#include "shogi_engine.h"
#include "GameTree.h"

static std::string moveToString(const shogi::engine::Move& move) {
  static const std::string pieceSymbols[14] = {
      "p", "l", "n", "s", "g", "b", "r", "P", "L", "N", "S", "G", "B", "R"};
  std::string moveString = "";
  if (move.from >= shogi::engine::WHITE_PAWN_DROP) {
    moveString +=
        pieceSymbols[move.from - shogi::engine::WHITE_PAWN_DROP] + "*";
  } else {
    int fromFile = shogi::engine::squareToFile(
        static_cast<shogi::engine::Square>(move.from));
    int fromRank = shogi::engine::squareToRank(
        static_cast<shogi::engine::Square>(move.from));
    moveString += std::to_string(BOARD_DIM - fromFile) +
                  static_cast<char>('a' + fromRank);
  }
  int toFile =
      shogi::engine::squareToFile(static_cast<shogi::engine::Square>(move.to));
  int toRank =
      shogi::engine::squareToRank(static_cast<shogi::engine::Square>(move.to));
  moveString +=
      std::to_string(BOARD_DIM - toFile) + static_cast<char>('a' + toRank);
  if (move.promotion) {
    moveString += '+';
  }
  return moveString;
}

static std::string getBestMoveUSI(const shogi::engine::Board& board,
                                  bool isWhite,
                                  unsigned int maxDepth,
                                  unsigned int maxTime = 0) {
  shogi::engine::GameTree gameTree(board, isWhite, maxDepth);
  shogi::engine::Move bestMove = gameTree.FindBestMove();
  return moveToString(bestMove);
}

BSTR getAllLegalMoves(const char* SFENstring) {
  bool isWhite;
  shogi::engine::Board board =
      shogi::engine::Board::FromSFEN(SFENstring, isWhite);
  std::vector<shogi::engine::Move> moves =
      shogi::engine::GameTree::GetAllMovesFrom(board, isWhite);
  std::string movesString = "";
  for (const auto& move : moves) {
    movesString += moveToString(move) + '|';
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
  shogi::engine::GameTree gameTree(board, isWhite, maxDepth);
  shogi::engine::Move bestMove = gameTree.FindBestMove();
  std::string bestMoveString = moveToString(bestMove);
  std::wstring bestMoveWstring(bestMoveString.begin(), bestMoveString.end());
  return SysAllocString(bestMoveWstring.c_str());
}
