#pragma once

#include "command_base.hpp"
#include "result_base.hpp"

namespace shogi::engine::command {

/// @brief Switch the debug mode of the engine on and off. In debug mode the
/// engine should send additional infos to the GUI, e.g. with the info string
/// command, to help debugging, e.g. the commands that the engine has received
/// etc. This mode should be switched off by default and this command can be
/// sent any time, also when the engine is thinking. (USI 5.3)
class Debug : public CommandBase {
 public:
  enum DebugEnabled { On, Off };

  Debug(DebugEnabled enabled) : _enabled{enabled} {}

  void execute(Instance& instance) override {};

 private:
  DebugEnabled _enabled;
};
}  // namespace shogi::engine::command