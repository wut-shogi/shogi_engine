#include "interface.hpp"
#include "result/usiok.hpp"

namespace shogi {
namespace engine {

void interface::accept_input(const std::string& input) {
  std::optional<command::CommandPtr> command = _parser.parse(input);
  
  if (command.has_value()) {
    //_invoker.post_command(std::move(command.value()));
  }

  return;
}

std::optional<result::ResultPtr> interface::try_get_result() {
  return std::nullopt;
}

result::ResultPtr interface::await_result() {
  return std::make_unique<result::usiok>();
}

}  // namespace engine
}  // namespace shogi