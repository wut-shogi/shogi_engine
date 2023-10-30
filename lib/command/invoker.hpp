#pragma once

#include <memory>
#include "command_base.hpp"
#include "../result/result_base.hpp"

namespace shogi {
namespace engine {
namespace command {

class invoker {
private:
 int _tasks_queue; // TODO: type

 public:
  void post_command(CommandPtr command);
  result::ResultPtr get_result();
};

}  // namespace command
}  // namespace engine
}  // namespace shogi