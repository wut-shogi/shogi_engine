#pragma once

#include "command_base.hpp"
#include "result_base.hpp"

namespace shogi::engine::command {

/// @brief The user has played the expected move. This will be sent if the
/// engine was told to ponder on the same move the user has played. The engine
/// should continue searching but switch from pondering to normal search.
/// (USI 5.3)
class PonderHit : public CommandBase {
 public:
  void execute(Instance& instance) override {}
};
}  // namespace shogi::engine::command