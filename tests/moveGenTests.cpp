#include <gtest/gtest.h>
#include "engine/lookUpTables.h"
#include "engine/search.h"

class MoveGenTests : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { shogi::engine::LookUpTables::CPU::init(); }
  static void TearDownTestSuite() {
    shogi::engine::LookUpTables::CPU::cleanup();
  }
};

TEST_F(MoveGenTests, PerftFromStartPos) {
  shogi::engine::Board startingBoard = shogi::engine::Boards::STARTING_BOARD();
  uint64_t nodesCount =
      shogi::engine::SEARCH::perftCPU<false>(startingBoard, 3, false);

  EXPECT_EQ(nodesCount, 25470);
}