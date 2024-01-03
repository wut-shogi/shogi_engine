#include <gtest/gtest.h>
#include "engine/USIconverter.h"
using namespace shogi::engine;

static void assertBoardsEqual(const Board& board1, const Board& board2) {
  for (int i = 0; i < BB::Type::SIZE; i++) {
    ASSERT_TRUE(board1[static_cast<BB::Type>(i)][TOP] ==
                board2[static_cast<BB::Type>(i)][TOP]);
    ASSERT_TRUE(board1[static_cast<BB::Type>(i)][MID] ==
                board2[static_cast<BB::Type>(i)][MID]);
    ASSERT_TRUE(board1[static_cast<BB::Type>(i)][BOTTOM] ==
                board2[static_cast<BB::Type>(i)][BOTTOM]);
  }
}

TEST(USIConversion, SFENToBoard) {
  std::string SFENstring =
      "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
  bool isWhite;
  Board board = SFENToBoard(SFENstring, isWhite);
  Board startingBoard = Boards::STARTING_BOARD();
  assertBoardsEqual(board, startingBoard);
  ASSERT_FALSE(isWhite);
}


TEST(USIConversion, BoardToSFEN) {
  Board board = Boards::STARTING_BOARD();
  bool isWhite = false;
  std::string SFENstring =
      "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b -";Board startingBoard = Boards::STARTING_BOARD();
  ASSERT_EQ(BoardToSFEN(board, isWhite), SFENstring);

  SFENstring = "8l/1l+R2P3/p2pBG1pp/kps1p4/Nn1P2G2/P1P1P2PP/1PS6/1KSG3+r1/LN2+p3L w Sbgn3p";
  board = SFENToBoard(SFENstring, isWhite);
  ASSERT_EQ(BoardToSFEN(board, isWhite), SFENstring);

}

TEST(USIConversion, MoveToUSI) {
  Move move;
  move.from = A9;
  move.to = A8;
  move.promotion = 0;
  ASSERT_TRUE(MoveToUSI(move) == "9a8a");
  move.from = C1;
  move.to = G9;
  move.promotion = 1;
  ASSERT_TRUE(MoveToUSI(move) == "1c9g+");
  move.from = B2;
  move.to = I6;
  move.promotion = 0;
  ASSERT_TRUE(MoveToUSI(move) == "2b6i");
  move.from = I1;
  move.to = A1;
  move.promotion = 1;
  ASSERT_TRUE(MoveToUSI(move) == "1i1a+");
  move.from = WHITE_LANCE_DROP;
  move.to = B5;
  move.promotion = 0;
  ASSERT_TRUE(MoveToUSI(move) == "L*5b");
  move.from = BLACK_LANCE_DROP;
  move.to = B5;
  move.promotion = 0;
  ASSERT_TRUE(MoveToUSI(move) == "L*5b");
}

static bool moveEqual(Move move1, Move move2) {
  return move1.from == move2.from && move1.to == move2.to &&
         move1.promotion == move2.promotion;
}

TEST(USIConversion, USIToMove) {
  Move move;
  move.from = A9;
  move.to = A8;
  move.promotion = 0;
  ASSERT_TRUE(moveEqual(move,USIToMove("9a8a", false)));
  move.from = C1;
  move.to = G9;
  move.promotion = 1;
  ASSERT_TRUE(moveEqual(move,USIToMove("1c9g+", false)));
  move.from = B2;
  move.to = I6;
  move.promotion = 0;
  ASSERT_TRUE(moveEqual(move,USIToMove("2b6i", false)));
  move.from = I1;
  move.to = A1;
  move.promotion = 1;
  ASSERT_TRUE(moveEqual(move,USIToMove("1i1a+", false)));
  move.from = WHITE_LANCE_DROP;
  move.to = B5;
  move.promotion = 0;
  ASSERT_TRUE(moveEqual(move,USIToMove("L*5b", true)));
  move.from = BLACK_LANCE_DROP;
  move.to = B5;
  move.promotion = 0;
  ASSERT_TRUE(moveEqual(move,USIToMove("L*5b", false)));
}
