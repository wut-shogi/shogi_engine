#include <iostream>
#include "../include/engine.h"
#include "search.h"
shogi::engine::Board Engine::board =
    shogi::engine::Boards::STARTING_BOARD();
bool Engine::isWhite = false;
uint16_t Engine::depth = 5;
uint32_t Engine::time = 0;
int Engine::gpuCount = 1;

void Engine::SetDepth(uint16_t depth) {
  Engine::depth = depth;
}
void Engine::SetPosition(const std::string& SFENstring) {
  Engine::board = shogi::engine::SFENToBoard(SFENstring, isWhite);
}
void Engine::SetTime(uint32_t time) {
  Engine::time = time;
}
void Engine::SetGPUCount(int gpuCount) {
  Engine::gpuCount = gpuCount;
}


void Engine::perftGPU() {
  shogi::engine::SEARCH::setDeviceCount(gpuCount);
  shogi::engine::SEARCH::init();
  shogi::engine::SEARCH::perft<true>(board, isWhite, depth, shogi::engine::SEARCH::GPU);
  shogi::engine::SEARCH::cleanup();
}
void Engine::perftCPU() {
  shogi::engine::SEARCH::init();
  shogi::engine::SEARCH::perft<true>(board, isWhite, depth, shogi::engine::SEARCH::CPU);
  shogi::engine::SEARCH::cleanup();
}
void Engine::getBestMoveGPU() {
  shogi::engine::SEARCH::setDeviceCount(gpuCount);
  shogi::engine::SEARCH::init();
  shogi::engine::SEARCH::GetBestMove(board, isWhite, depth, time,
                                     shogi::engine::SEARCH::GPU);
  shogi::engine::SEARCH::cleanup();
}
void Engine::getBestMoveCPU() {
  shogi::engine::SEARCH::init();
  shogi::engine::SEARCH::GetBestMove(board, isWhite, depth, time,
                                     shogi::engine::SEARCH::CPU);
  shogi::engine::SEARCH::cleanup();
}
