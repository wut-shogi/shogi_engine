#include "interface.hpp"
#include "result/usiok.hpp"

namespace shogi::engine {

void Interface::acceptInput(const std::string& input) {
  std::optional<command::CommandPtr> command = _parser.parse(input);

  if (command.has_value()) {
    //_instance.post_command(std::move(command.value()));
  }
}

std::optional<result::ResultPtr> Interface::tryGetResult() {
  return std::nullopt;
}

result::ResultPtr Interface::awaitResult() {
  return std::make_unique<result::UsiOk>();
}

}  // namespace shogi::engine