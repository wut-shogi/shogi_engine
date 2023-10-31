#pragma once

#include <thread>
#include "command/command_base.hpp"
#include "option/options.hpp"
#include "result/readyok.hpp"
#include "result/result_base.hpp"
#include "utils/thread_safe_queue.hpp"

namespace shogi::engine {

/// @brief Engine instance. This is the class that actually processes
/// requests through calls to its methods.
class Instance {
  option::Options _options;
  utils::SafeQueue<result::ResultPtr> _results;

 public:
  Instance(Instance&) = delete;
  Instance(Instance&&) = default;

  Instance& operator=(Instance&) = delete;
  Instance& operator=(Instance&&) = default;
  Instance() = default;
  ~Instance() = default;
  void postCommand(command::CommandPtr command);
  result::ResultPtr getResult();
};
}  // namespace shogi::engine