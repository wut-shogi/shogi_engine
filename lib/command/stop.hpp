#pragma once

#include "command_base.hpp"
#include "result_base.hpp"

namespace shogi::engine::command {

/// @brief Stop calculating as soon as possible. Don't forget the bestmove and
/// possibly the ponder token when finishing the search. (USI 5.3)
class Stop : public CommandBase {
 public:
  void execute(Instance& instance) override {}
};
}  // namespace shogi::engine::command