#pragma once

#include <memory>
#include <optional>
#include <string>
#include "instance.hpp"
#include "parser.hpp"
#include "result/result_base.hpp"

namespace shogi::engine {

/// @brief Interface to engine instance. Used to communicate with engine
/// according to Universal Shogi Interface. Monitores engine state and delegates
/// necessary calculations to the engine instance.
class Interface {
 private:
  Instance _instance;
  Parser _parser;

 public:
  Interface(Instance&& instance) : _instance{std::move(instance)} {}

  Interface() = delete;
  ~Interface() = default;
  Interface(Interface&) = delete;
  Interface(Interface&&) = default;
  Interface& operator=(Interface&) = delete;
  Interface& operator=(Interface&&) = default;

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
  void acceptInput(const std::string& input);

  /// @brief Immediate return.
  /// @return Optional of first calculated result or nothing, if there is no
  /// calculated results
  std::optional<result::ResultPtr> tryGetResult();

  /// @brief Blocking return.
  /// @return Result pointer of first calculated result.
  result::ResultPtr awaitResult();
};
}  // namespace shogi::engine