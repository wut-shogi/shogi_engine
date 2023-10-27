#include "interface.hpp"
#include "result/usiok.hpp"

namespace shogi {
namespace engine {
void interface::accept_command(const std::string& command) {
  
  return;
}

std::optional<ResultPtr> interface::try_get_result() {
  return std::nullopt;
}

ResultPtr interface::await_result() {
  return std::make_unique<result::usiok>();
}

}  // namespace engine
}  // namespace shogi