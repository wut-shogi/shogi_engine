#include "parser.hpp"
#include "command/usi.hpp"

namespace shogi::engine {
std::optional<command::CommandPtr> Parser::parse(const std::string& input) {
  return std::optional(std::make_unique<command::Usi>());
};
}  // namespace shogi::engine