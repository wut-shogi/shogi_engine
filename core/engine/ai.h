#pragma once
#include "MoveGen.h"

namespace shogi {
namespace engine {
struct TreeLevel {
  Board* boardsArray;
  uint32_t length;
  uint32_t* childrenRange;
  Move* movesArray;
  bool isWhite;
  int depth;
};

TreeLevel buildNextLevel(TreeLevel& level);
}  // namespace engine
}  // namespace shogi
