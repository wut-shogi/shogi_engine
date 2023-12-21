#include <gtest/gtest.h>
#include "engine/CPUsearchHelpers.h"
#include "engine/lookUpTables.h"

class MoveGenTests : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { shogi::engine::LookUpTables::CPU::init(); }
  static void TearDownTestSuite() {
    shogi::engine::LookUpTables::CPU::cleanup();
  }
};

TEST_F(MoveGenTests, PerftFromStartPos) {
  std::vector<shogi::engine::Move> moves;
  shogi::engine::Board startingBoard = shogi::engine::Boards::STARTING_BOARD();
  uint64_t nodesCount =
      shogi::engine::CPU::perft<true>(startingBoard, 3, moves, false);

  EXPECT_EQ(nodesCount, 25470);
}