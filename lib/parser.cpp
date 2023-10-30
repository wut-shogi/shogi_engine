#include "parser.hpp"
#include "command/usi.hpp"

namespace shogi {
namespace engine {
std::optional<command::CommandPtr> parser::parse(const std::string& input){
    return std::optional(std::make_unique<command::usi>());
};
}
}  // namespace shogi