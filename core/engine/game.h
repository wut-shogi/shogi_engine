#pragma once
#include <chrono>
#include <vector>
#include "lookUpTables.h"
#include "moveGenHelpers.h"
#include "search.h"

namespace shogi {
namespace engine {

class GameSimulator {
 public:
  GameSimulator(uint16_t maxDepth, uint16_t maxTime, SEARCH::SearchType type)
      : type(type), maxDepth(maxDepth), maxTime(maxTime) {}
  void Run() {
    bool result = SEARCH::init();
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
        Move bestMove = SEARCH::GetBestMove(board, isWhite, maxDepth, maxTime, type);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        /*std::cout << "Time: " << duration.count() << " ms" << std::endl;
        printf("Best Move: %s\n", moveToUSI(bestMove));*/
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
  SEARCH::SearchType type;
  uint16_t maxDepth;
  uint16_t maxTime;
};

}  // namespace engine
}  // namespace shogi
