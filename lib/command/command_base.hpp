#pragma once

#include <memory>
#include "../result/result_base.hpp"
#include "../utils/thread_safe_queue.hpp"

namespace shogi {
namespace engine {

class instance;

namespace command {

class command_base;

using CommandPtr = std::unique_ptr<command_base>;

class command_base {
 public:
  virtual void execute(instance& instance) = 0;
};

}  // namespace command
}  // namespace engine
}  // namespace shogi