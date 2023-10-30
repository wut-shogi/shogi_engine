#pragma once

#include <thread>
#include "command/command_base.hpp"
#include "option/options.hpp"
#include "result/readyok.hpp"
#include "result/result_base.hpp"
#include "utils/thread_safe_queue.hpp"

namespace shogi {
namespace engine {

/// @brief Engine instance. This is the class that actually processes
/// requests through calls to its methods.
class instance {
 protected:
  option::options _options;
  utils::thread_safe_queue<result::ResultPtr> _results;

 public:
  instance(instance&) = delete;
  instance(instance&&);
  instance();
  void post_command(command::CommandPtr command);
  result::ResultPtr get_result();
};
}  // namespace engine
}  // namespace shogi