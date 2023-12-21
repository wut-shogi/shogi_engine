#pragma once
#include <vector>
#include "Board.h"
#include "CPUsearchHelpers.h"

namespace shogi {
namespace engine {
namespace search {

bool init();

Move GetBestMove(const Board& board,
                 bool isWhite,
                 uint16_t depth,
                 uint16_t maxDepth);
}  // namespace search
}  // namespace engine
}  // namespace shogi
