#pragma once
#include "Board.h"

namespace shogi {
namespace engine {
RUNTYPE int16_t evaluate(const Board& board, bool isWhite);
}
}  // namespace shogi