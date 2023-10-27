#pragma once
#include "result_base.hpp"
#include <sstream>

namespace shogi {
namespace engine {
namespace result {
class id : public result_base {
 private:
  static constexpr std::string_view _name = "wut shogi";
  static constexpr std::string_view _author = "us";

 public:
  virtual std::string to_string() const override {
    std::ostringstream out;
    out << "id name " << _name << "\n id author " << _author << "\n";
    return out.str();
  }
};
}  // namespace result
}  // namespace engine
}  // namespace shogi