#pragma once

#include "command_base.hpp"
#include "result_base.hpp"

namespace shogi {
namespace engine {
namespace command {

/// @brief  Start calculating on the current position set up with the position
/// command. There are a number of commands that can follow this command, all
/// will be sent in the same string. If one command is not sent its value should
/// be interpretedas if it would not influence the search. (USI 5.3)
class go : public command_base {
 public:
  virtual void execute(instance& instance) override{
      // instance.post_result(std::move(result));
  };
};
}  // namespace command
}  // namespace engine
}  // namespace shogi