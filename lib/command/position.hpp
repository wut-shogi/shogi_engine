#pragma once

#include "command_base.hpp"
#include "result_base.hpp"

namespace shogi {
namespace engine {
namespace command {

/// @brief Set up the position described in sfenstring on the internal board and
/// play the moves on the internal board. If the game was played from the start
/// position, the string startpos will be sent.
///
/// Note: If this position is from a different game than the last position sent
/// to the engine, the GUI should have sent a usinewgame inbetween. (USI 5.3)
class isready : public command_base {
 public:
  virtual void execute(invoker& invoker) override;
};
}  // namespace command
}  // namespace engine
}  // namespace shogi