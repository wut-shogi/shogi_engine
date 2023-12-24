#pragma once
#include "Board.h"

namespace shogi {
namespace engine {
__host__ __device__ int16_t evaluate(const Board& board);
}
}  // namespace shogi