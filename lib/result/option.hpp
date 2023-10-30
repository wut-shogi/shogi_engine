#pragma once
#include <sstream>
#include "result_base.hpp"

namespace shogi {
namespace engine {
namespace result {
/// @brief  This command tells the GUI which parameters can be changed in the
/// engine. This should be sent once at engine startup after the usi and the id
/// commands if any parameter can be changed in the engine. The GUI should parse
/// this and build a dialog for the user to change the settings. Note that not
/// every option should appear in this dialog, as some options like USI_Ponder,
/// USI_AnalyseMode, etc. are better handled elsewhere or are set automatically.
///
/// If the user wants to change some settings, the GUI will send a setoption
/// command to the engine.
///
/// Note that the GUI need not send the setoption command when starting the
/// engine for every option if it doesn't want to change the default value. For
/// all allowed combinations see the examples below, as some combinations of
/// this tokens don't make sense.
///
/// One string will be sent for each parameter. (USI 5.3)
class option : public result_base {
 public:
  virtual std::string to_string() const override;
};
}  // namespace result
}  // namespace engine
}  // namespace shogi