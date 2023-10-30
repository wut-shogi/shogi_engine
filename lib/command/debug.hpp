#pragma once

#include "command_base.hpp"
#include "result_base.hpp"

namespace shogi {
namespace engine {
namespace command {

/// @brief Switch the debug mode of the engine on and off. In debug mode the
/// engine should send additional infos to the GUI, e.g. with the info string
/// command, to help debugging, e.g. the commands that the engine has received
/// etc. This mode should be switched off by default and this command can be
/// sent any time, also when the engine is thinking. (USI 5.3)
class debug : public command_base {
 public:
  enum debug_enabled { ON, OFF };

  debug(debug_enabled enabled) : _enabled{enabled} {}

  virtual void execute(instance& instance) override{
      // instance.post_result(std::move(result));
  };

 private:
  debug_enabled _enabled;
};
}  // namespace command
}  // namespace engine
}  // namespace shogi