#pragma once

#include "types.hpp"

namespace shogi::engine::option {
struct Options {
  Spin<1, 0, 2> USI_Hash;
  Check<false> USI_Ponder;
  Check<false> USI_OwnBook;
  Spin<1, 1, 10> USI_MultiPV;
  Check<false> USI_ShowCurrLine;
  Check<false> USI_ShowRefutations;
  Check<false> USI_LimitStrength;
  Spin<-10, 1, 10> USI_Strength;
  Check<false> USI_AnalyseMode;
};
}  // namespace shogi::engine::option