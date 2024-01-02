#pragma once

#include <memory>
#include "../result/result_base.hpp"
#include "../utils/thread_safe_queue.hpp"

namespace shogi::engine {

class Instance;

namespace command {

class CommandBase;

using CommandPtr = std::unique_ptr<CommandBase>;

class CommandBase {
 public:
  virtual void execute(Instance& instance) = 0;
  virtual ~CommandBase() = default;
  CommandBase() = default;
  CommandBase(CommandBase&) = default;
  CommandBase(CommandBase&&) = default;
  CommandBase& operator=(const CommandBase&) = default;
  CommandBase& operator=(CommandBase&&) = default;
};

}  // namespace command
}  // namespace shogi::engine