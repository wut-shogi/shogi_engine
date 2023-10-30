#pragma once

#include <memory>
#include <optional>
#include <string>
#include "instance.hpp"
#include "parser.hpp"
#include "result/result_base.hpp"

namespace shogi {
namespace engine {

/// @brief Interface to engine instance. Used to communicate with engine
/// according to Universal Shogi Interface. Monitores engine state and delegates
/// necessary calculations to the engine instance.
class interface {
 private:
  instance _instance;
  parser _parser;

 public:
  interface(instance&& instance) : _instance{instance} {}

  interface() = delete;
  interface(interface&) = delete;
  interface& operator=(interface) = delete;

  /// @brief Accepts input commands according to Universal Shogi Interface.
  /// @param input Command string. It should be a single ASCII line,
  /// f.e. passed from stdin.
  /// @note
  /// - It returns without actually performing requested actions (USI 5.1)
  /// @note
  /// - Invalid tokens are ignored and the rest of input is processed
  /// independently (USI 5.1)
  /// @note
  /// - Invalid input in given context is also ignored (USI 5.1)
  void accept_input(const std::string& input);

  /// @brief Immediate return.
  /// @return Optional of first calculated result or nothing, if there is no
  /// calculated results
  std::optional<result::ResultPtr> try_get_result();

  /// @brief Blocking return.
  /// @return Result pointer of first calculated result.
  result::ResultPtr await_result();
};
}  // namespace engine
}  // namespace shogi