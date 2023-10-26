#pragma once

#include <iostream>
#include <string>

namespace shogi {
namespace engine {
namespace result {

class result_base {
 public:
  virtual std::string to_string() const = 0;

  friend std::ostream& operator<<(std::ostream& out, const result_base& res) {
    return out << res.to_string();
  }
};

}  // namespace result
}  // namespace engine
}  // namespace shogi