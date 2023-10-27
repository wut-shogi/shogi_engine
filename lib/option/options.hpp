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
    check<false> USI_ShowCurrLine;
    check<false> USI_ShowRefutations;
    check<false> USI_LimitStrength;
    spin<-10, 1, 10> USI_Strength;
    check<false> USI_AnalyseMode;
};
}  // namespace option
}  // namespace engine
}  // namespace shogi