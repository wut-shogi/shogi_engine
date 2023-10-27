#pragma once

#include <string>
#include "command/command_base.hpp"
#include <optional>

namespace shogi {
namespace engine {
class parser {
 public:
  std::optional<command::CommandPtr> parse(const std::string& input);
};
}  // namespace engine
}  // namespace shogi