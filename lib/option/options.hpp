#pragma once

#include "types.hpp"

namespace shogi {
namespace engine {
namespace option {
struct options {
    spin<1, 0, 2> USI_Hash;
    check<false> USI_Ponder;
    check<false> USI_OwnBook;
    spin<1, 1, 10> USI_MultiPV;
    spin<1, 0, 2> USI_ShowCurrLine;
};
}  // namespace option
}  // namespace engine
}  // namespace shogi