#include "instance.hpp"
#include <thread>
#include "result/usiok.hpp"

namespace shogi {
namespace engine {

void instance::post_command(command::CommandPtr command) {}

result::ResultPtr instance::get_result() {
  return std::make_unique<result::usiok>();
}
}  // namespace engine
}  // namespace shogi