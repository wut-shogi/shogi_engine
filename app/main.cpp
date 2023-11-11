#include <iostream>
#include <memory>
#include <shogi/engine.hpp>

int main(int argc, char* argv[]) {
  std::cout << "Hello world!\n";
  shogi::engine::Interface interface {
    shogi::engine::Instance {}
  };

  std::cout << *(interface.awaitResult()) << std::endl;
  return 0;
}
