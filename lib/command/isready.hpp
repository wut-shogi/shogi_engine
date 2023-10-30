#pragma once

#include "command_base.hpp"
#include "readyok.hpp"
#include "result_base.hpp"

namespace shogi {
namespace engine {
namespace command {

/// @brief This is used to synchronize the engine with the GUI. When the GUI has
/// sent a command or multiple commands that can take some time to complete,
/// this command can be used to wait for the engine to be ready again or to ping
/// the engine to find out if it is still alive. This command is also required
/// once before the engine is asked to do any search to wait for the engine to
/// finish initializing. This command must always be answered with readyok and
/// can be sent also when the engine is calculating in which case the engine
/// should also immediately answer with readyok without stopping the search.
/// (USI 5.3)
class isready : public command_base {
 public:
  virtual void execute(invoker& invoker) override {
    result::ResultPtr result = std::make_unique<result::readyok>();
    // invoker.post_result(std::move(result));
  };
};
}  // namespace command
}  // namespace engine
}  // namespace shogi