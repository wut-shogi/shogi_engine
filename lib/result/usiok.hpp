#pragma once

#include "result_base.hpp"

namespace shogi {
namespace engine {
namespace result {

class usiok : public result_base {
  virtual std::string to_string() const override { return "usiok\n"; }
};
}  // namespace result
}  // namespace engine
}  // namespace shogi