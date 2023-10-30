#pragma once

#include "command_base.hpp"
#include "result_base.hpp"

namespace shogi {
namespace engine {
namespace command {

/// @brief This is sent to the engine when the user wants to change the internal
/// parameters of the engine. For the button type no value is needed. One string
/// will be sent for each parameter and this will only be sent when the engine
/// is waiting. The name and value of the option in <id> should not be case
/// sensitive and can not include spaces. (USI 5.3)
class setoption : public command_base {
 public:
  virtual void execute(invoker& invoker) override{
      // invoker.post_result(std::move(result));
  };
};
}  // namespace command
}  // namespace engine
}  // namespace shogi