#pragma once

#include "command_base.hpp"
#include "result_base.hpp"

namespace shogi::engine::command {

/// @brief (Shogidogoro) Informs the engine that the game has ended with the
/// specified result, from the engine's own point or view. (USI 5.3)
class GameOver : public CommandBase {
 public:
  void execute(Instance& instance) override {}
};
}  // namespace shogi::engine::command