#pragma once
#include "result_base.hpp"

namespace shogi {
namespace engine {
namespace result {
class id : public result_base {
 private:
  std::string _name = "shogi wut";
  std::string _author = "us";

 public:
  virtual std::string to_string() const override {
    // TODO: make it nicer
    return "id name " + _name + "\n id author " + _author + "\n";
  }
};
}  // namespace result
}  // namespace engine
}  // namespace shogi