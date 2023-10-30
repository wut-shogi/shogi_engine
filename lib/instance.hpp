#pragma once

#include "result/readyok.hpp"

namespace shogi {
namespace engine {
/// @brief Engine instance. This is the class that actually processes
/// requests through calls to its methods.
class instance {
 public:

  result::readyok isready();

  template <class TName>
  void setoption(TName name);

  template <class TName, typename TValue>
  void setoption(TName name, TValue value);

  void usinewgame();
};
}  // namespace engine
}  // namespace shogi