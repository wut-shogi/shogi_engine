#pragma once
#include <vector>
#include "Board.h"
#include "CPUsearchHelpers.h"

namespace shogi {
namespace engine {
namespace SEARCH {

bool init();

void cleanup();

Move GetBestMove(const Board& board, bool isWhite, uint16_t maxDepth);

Move GetBestMoveAlphaBeta(const Board& board, bool isWhite, uint16_t maxDepth);

int16_t alphaBeta(Board& board,
                  bool isWhite,
                  uint16_t depth,
                  int16_t alpha,
                  int16_t beta,
                  std::vector<uint32_t>& nodesSearched);

}  // namespace SEARCH
}  // namespace engine
}  // namespace shogi
