#include "instance.hpp"
#include <thread>
#include "result/usiok.hpp"

namespace shogi::engine {

void Instance::postCommand(command::CommandPtr command) {
  std::thread thread{[command = std::move(command), &instance = *this]() {
    command->execute(instance);
  }};
  thread.detach();
}

result::ResultPtr Instance::getResult() {
  return std::make_unique<result::UsiOk>();
}
}  // namespace shogi::engine