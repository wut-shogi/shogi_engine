#pragma once

#include "../instance.hpp"
#include "../result/result_base.hpp"
#include "../result/usiok.hpp"
#include "command_base.hpp"

namespace shogi {
namespace engine {

class instance;

namespace command {

/// @brief Tell engine to use the USI (universal shogi interface). This will be
/// sent once as a first command after program boot to tell the engine to switch
/// to USI mode. After receiving the usi command the engine must identify itself
/// with the id command and send the option commands to tell the GUI which
/// engine settings the engine supports. After that, the engine should send
/// usiok to acknowledge the USI mode. If no usiok is sent within a certain time
/// period, the engine task will be killed by the GUI. (USI 5.3)
class usi : public command_base {
 public:
  virtual void execute(instance& instance) override{
      // command_base::result_queue(instance).push_back(
      //    std::make_unique<result::usiok>());
  };
};
}  // namespace command
}  // namespace engine
}  // namespace shogi