#pragma once
#include "invoker.hpp"

namespace shogi {
namespace engine {
namespace command {
class command_base;

using CommandPtr = std::unique_ptr<command_base>;

class command_base {
 public:
  virtual void execute(invoker& invoker) = 0;
};
}  // namespace command
}  // namespace engine
}  // namespace shogi