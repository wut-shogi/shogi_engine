#pragma once

#include <memory>

namespace shogi {
namespace engine {
class invoker;

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