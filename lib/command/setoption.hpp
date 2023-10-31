#pragma once

#include "command_base.hpp"
#include "result_base.hpp"

namespace shogi::engine::command {

/// @brief This is sent to the engine when the user wants to change the internal
/// parameters of the engine. For the button type no value is needed. One string
/// will be sent for each parameter and this will only be sent when the engine
/// is waiting. The name and value of the option in <id> should not be case
/// sensitive and can not include spaces. (USI 5.3)
class SetOption : public CommandBase {
 public:
  void execute(Instance& instance) override {}
};
}  // namespace shogi::engine::command