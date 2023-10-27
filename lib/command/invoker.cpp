#include "invoker.hpp"

namespace shogi {
namespace engine {
namespace command {
result::ResultPtr invoker::get_result() {

}

void invoker::post_command(CommandPtr command) {
    command->execute(*this);
}

}  // namespace command
}  // namespace engine
}  // namespace shogi