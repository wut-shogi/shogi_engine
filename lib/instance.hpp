#pragma once

#include "result/readyok.hpp"
#include "command/command_base.hpp"
#include "option/options.hpp"

namespace shogi {
namespace engine {
/// @brief Engine instance. This is the class that actually processes
/// requests through calls to its methods.
class instance {

  option::options _options;

public:
  void post_command(command::CommandPtr command);
  result::ResultPtr get_result();
};
}  // namespace engine
}  // namespace shogi