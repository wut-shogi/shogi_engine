#pragma once
#include "../engine/Board.h"

class Engine {
 public:
  static void SetDepth(uint16_t depth);
  static void SetPosition(const std::string& SFENstring);
  static void SetTime(uint32_t time);
  static void SetGPUCount(int gpuCount);

  static void perftGPU();
  static void perftCPU();
  static void getBestMoveGPU();
  static void getBestMoveCPU();

 private:
  static shogi::engine::Board board;
  static bool isWhite;
  static uint16_t depth;
  static uint32_t time;
  static int gpuCount;
};