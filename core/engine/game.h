#pragma once
#include <vector>
#include <chrono>
#include "moveGenHelpers.h"
#include "lookUpTables.h"
#include "gameTree.h"

namespace shogi {
namespace engine {

class GameSimulator {
 public:
  void Run() {
    bool result = search::init();
    if (!result) {
      std::cout << "init Error" << std::endl;
    }
    board = Boards::STARTING_BOARD();
    isWhite = false;
    print_Board(board);
    std::string command;
    while (std::cin >> command) {
      if (command == "q") {
        break;
      } else if (command == "n") {
        auto start = std::chrono::high_resolution_clock::now();
        Move bestMove =
            search::GetBestMove(board, isWhite, 0, 5);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Time: " << duration.count() << " ms" << std::endl;
        makeMove(board, bestMove);
        print_Board(board);
        isWhite = !isWhite;
      }
    }
  }

 private:
  Board board;
  bool isWhite;
  uint8_t* d_Buffer;
  uint32_t d_BufferSize;
};

}  // namespace engine
}  // namespace shogi
