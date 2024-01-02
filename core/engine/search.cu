#include <thrust/extrema.h>
#include <atomic>
#include <chrono>
#include <future>
#include "CPUsearchHelpers.h"
#include "GPUsearchHelpers.h"
#include "evaluation.h"
#include "lookUpTables.h"
#include "search.h"
#include "USIconverter.h"

namespace shogi {
namespace engine {
namespace SEARCH {

std::atomic<bool> terminateSearch(false);

uint8_t* d_Buffer;
uint32_t d_BufferSize = 0;

bool init() {
  try {
    LookUpTables::CPU::init();
    LookUpTables::GPU::init();
    size_t total = 0, free = 0;
    cudaMemGetInfo(&free, &total);
    if (free == 0)
      return false;
    d_BufferSize = (free / 4) * 4;
    cudaError_t error = cudaMalloc((void**)&d_Buffer, d_BufferSize);
    if (error != cudaSuccess)
      return false;
  } catch (...) {
    return false;
  }
  return true;
}

void cleanup() {
  LookUpTables::CPU::cleanup();
  LookUpTables::GPU::cleanup();
  if (d_BufferSize > 0)
    cudaFree(d_Buffer);
}

void IterativeDeepeningSearch(Move& outBestMove,
                              const Board& board,
                              bool isWhite,
                              uint16_t maxDepth,
                              Move (*GetBestMove)(const Board&,
                                                  bool,
                                                  uint16_t)) {
  uint16_t minDepth = std::min((uint16_t)3, maxDepth);
  for (int depth = minDepth; depth <= maxDepth; depth++) {
    Move bestMove = GetBestMove(board, isWhite, depth);
    if (!terminateSearch)
      outBestMove = bestMove;
    else
      return;
  }
}

int16_t alphaBeta(Board& board,
                  bool isWhite,
                  uint16_t depth,
                  int16_t alpha,
                  int16_t beta,
                  std::vector<uint32_t>& nodesSearched) {
  if (terminateSearch)
    return 0;
  CPU::MoveList moves(board, isWhite);
  if (moves.size() == 0) {
    return isWhite ? INT16_MIN : INT16_MAX;
  }
  if (depth == 0) {
    return evaluate(board, isWhite);
  }
  Board oldBoard = board;
  int16_t result = 0;
  if (isWhite) {
    result = INT16_MIN;
    for (const Move& move : moves) {
      makeMove(board, move);
      result =
          alphaBeta(board, !isWhite, depth - 1, alpha, beta, nodesSearched);
      board = oldBoard;
      nodesSearched[nodesSearched.size() - depth]++;
      if (result > alpha) {
        alpha = result;
      }
      if (alpha >= beta) {
        break;
      }
    }
    result = alpha;
  } else {
    for (const Move& move : moves) {
      makeMove(board, move);
      result =
          alphaBeta(board, !isWhite, depth - 1, alpha, beta, nodesSearched);
      board = oldBoard;
      nodesSearched[nodesSearched.size() - depth]++;
      if (result < beta) {
        beta = result;
      }
      if (alpha >= beta) {
        break;
      }
    }
    result = beta;
  }
  return result;
}

Move GetBestMoveCPU(const Board& board, bool isWhite, uint16_t maxDepth) {
  auto start = std::chrono::high_resolution_clock::now();
  CPU::MoveList rootMoves(board, isWhite);
  std::vector<uint32_t> nodesSearched(maxDepth, 0);
  CPU::MoveList moves(board, isWhite);
  if (moves.size() == 0) {
    return Move{0, 0, 0};
  }
  Move bestMove = *moves.begin();
  int16_t score = isWhite ? INT16_MIN : INT16_MAX;
  Board newBoard = board;
  for (const auto& move : moves) {
    makeMove(newBoard, move);
    int16_t result = alphaBeta(newBoard, !isWhite, maxDepth - 1, INT16_MIN,
                               INT16_MAX, nodesSearched);
    if (terminateSearch)
      return bestMove;
    newBoard = board;
    if ((isWhite && result > score) || (!isWhite && result < score)) {
      bestMove = move;
      score = result;
    }
  }

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "Generated: " << moves.size() << " positions on depth: " << 1
            << std::endl;
  for (int i = 1; i < nodesSearched.size(); i++) {
    std::cout << "Generated: " << nodesSearched[i]
              << " positions on depth: " << i + 1 << std::endl;
  }
  std::cout << " Best move found: " << MoveToUSI(bestMove) << std::endl;
  std::cout << "Time: " << duration.count() << " ms" << std::endl;
  return bestMove;
}

uint64_t countMovesCPU(Board& board, uint16_t depth, bool isWhite) {
  CPU::MoveList moves(board, isWhite);
  if (depth == 1)
    return moves.size();
  uint64_t moveCount = 0;
  Board oldBoard = board;
  for (const auto& move : moves) {
    MoveInfo moveReturnInfo = makeMove<true>(board, move);
    moveCount += countMovesCPU(board, depth - 1, !isWhite);
    // unmakeMove(board, move, moveReturnInfo);
    board = oldBoard;
  }
  return moveCount;
}

GPUBuffer::GPUBuffer(const Board& startBoard) {
  d_startBoard = (Board*)d_Buffer;
  cudaMemcpy(d_startBoard, &startBoard, sizeof(Board), cudaMemcpyHostToDevice);
  freeBegin = d_Buffer + sizeof(Board);
  freeEnd = d_Buffer + d_BufferSize;
}

Board* GPUBuffer::GetStartBoardPtr() {
  return d_startBoard;
}
bool GPUBuffer::ReserveMovesSpace(uint32_t size,
                                  int16_t movesPerBoard,
                                  Move*& outMovesPtr) {
  outMovesPtr = (Move*)freeBegin;
  freeBegin += size * movesPerBoard * sizeof(Move);
  return freeBegin < freeEnd;
}
void GPUBuffer::FreeMovesSpace(Move* moves) {
  freeBegin = (uint8_t*)moves;
}
bool GPUBuffer::ReserveOffsetsSpace(uint32_t size, uint32_t*& outOffsetsPtr) {
  uint32_t aligmentMismatch = (size_t)freeBegin % 4;
  if (aligmentMismatch != 0) {
    freeBegin += 4 - aligmentMismatch;
  }
  outOffsetsPtr = (uint32_t*)freeBegin;
  freeBegin += size * sizeof(uint32_t);
  return freeBegin < freeEnd;
}
void GPUBuffer::FreeOffsetsSpace(uint32_t* offsets) {
  freeBegin = (uint8_t*)offsets;
}
bool GPUBuffer::ReserveBitboardsSpace(uint32_t size,
                                      uint32_t*& outBitboardsPtr) {
  // It is temporary so we allocate it from the back so it can be freed easily
  freeEnd -= size * 3 * 3 * sizeof(uint32_t);
  outBitboardsPtr = (uint32_t*)freeEnd;
  return freeBegin < freeEnd;
}

void GPUBuffer::FreeBitboardsSpace(uint32_t size) {
  freeEnd = d_Buffer + d_BufferSize;
}

static const uint32_t maxProcessedSize = 5000;

// Converts moves to values
void minMaxGPU(Move* moves,
               uint32_t size,
               bool isWhite,
               uint16_t depth,
               uint16_t maxDepth,
               GPUBuffer& gpuBuffer,
               std::vector<uint32_t>& numberOfMovesPerDepth) {
  if (terminateSearch)
    return;
  if (depth == maxDepth) {
    // Evaluate moves
    GPU::evaluateBoards(size, isWhite, depth, gpuBuffer.GetStartBoardPtr(), moves,
                        (int16_t*)moves);
    numberOfMovesPerDepth[depth - 1] += size;
    return;
  }
  uint32_t processed = 0;
  uint32_t* offsets;
  uint32_t* bitboards;
  uint32_t* bestIndex;
  // Process by chunks
  while (processed < size) {
    uint32_t sizeToProcess = std::min(size - processed, maxProcessedSize);
    // Calculate offsets
    if (!gpuBuffer.ReserveOffsetsSpace(sizeToProcess + 1, offsets))
      printf("Err in ReserveOffsetsSpace\n");
    if (!gpuBuffer.ReserveBitboardsSpace(sizeToProcess, bitboards))
      printf("Err in ReserveBitboardsSpace\n");
    isWhite ? GPU::countWhiteMoves(sizeToProcess, depth,
                                   gpuBuffer.GetStartBoardPtr(), moves, size,
                                   processed, offsets, bitboards)
            : GPU::countBlackMoves(sizeToProcess, depth,
                                   gpuBuffer.GetStartBoardPtr(), moves, size,
                                   processed, offsets, bitboards);
    GPU::prefixSum(offsets, sizeToProcess + 1);
    uint32_t nextLayerSize = 0;
    cudaMemcpy(&nextLayerSize, offsets + sizeToProcess, sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    // Generate new moves
    Move* newMoves;

    if (!gpuBuffer.ReserveMovesSpace(nextLayerSize, depth + 1, newMoves))
      printf("Err in ReserveMovesSpace\n");
    isWhite ? GPU::generateWhiteMoves(sizeToProcess, depth,
                                      gpuBuffer.GetStartBoardPtr(), moves, size,
                                      processed, offsets, bitboards, newMoves)
            : GPU::generateBlackMoves(sizeToProcess, depth,
                                      gpuBuffer.GetStartBoardPtr(), moves, size,
                                      processed, offsets, bitboards, newMoves);
    gpuBuffer.FreeBitboardsSpace(sizeToProcess);
    // minmaxGPU(newLayer)
    minMaxGPU(newMoves, nextLayerSize, !isWhite, depth + 1, maxDepth, gpuBuffer,
              numberOfMovesPerDepth);
    bestIndex = offsets;
    // Gather values from new layer
    isWhite ? GPU::gatherValuesMax(sizeToProcess, depth, offsets,
                                   (int16_t*)newMoves,
                                   (int16_t*)(moves + processed), bestIndex)
            : GPU::gatherValuesMin(sizeToProcess, depth, offsets,
                                   (int16_t*)newMoves,
                                   (int16_t*)(moves + processed), bestIndex);
    gpuBuffer.FreeMovesSpace(newMoves);
    gpuBuffer.FreeOffsetsSpace(offsets);
    processed += sizeToProcess;
  }
  numberOfMovesPerDepth[depth - 1] += size;
}

Move GetBestMoveGPU(const Board& board, bool isWhite, uint16_t maxDepth) {
  auto start = std::chrono::high_resolution_clock::now();
  std::vector<uint32_t> numberOfMovesPerDepth(maxDepth, 0);
  GPUBuffer gpuBuffer(board);
  CPU::MoveList rootMoves(board, isWhite);
  if (rootMoves.size() == 0) {
    return Move{0, 0, 0};
  }
  Move bestMove = *rootMoves.begin();
  Move* d_moves;
  gpuBuffer.ReserveMovesSpace(rootMoves.size(), 1, d_moves);
  cudaMemcpy(d_moves, rootMoves.begin(), rootMoves.size() * sizeof(Move),
             cudaMemcpyHostToDevice);
  minMaxGPU(d_moves, rootMoves.size(), !isWhite, 1, maxDepth, gpuBuffer,
            numberOfMovesPerDepth);
  if (terminateSearch)
    return bestMove;
  std::vector<int16_t> h_values(rootMoves.size());
  cudaMemcpy(h_values.data(), d_moves, rootMoves.size() * sizeof(int16_t),
             cudaMemcpyDeviceToHost);
  size_t bestValueIdx =
      isWhite ? std::max_element(h_values.begin(), h_values.end()) -
                    h_values.begin()
              : std::min_element(h_values.begin(), h_values.end()) -
                    h_values.begin();
  bestMove = *(rootMoves.begin() + bestValueIdx);

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  numberOfMovesPerDepth[0] = rootMoves.size();
  for (int i = 0; i < numberOfMovesPerDepth.size(); i++) {
    std::cout << "Generated: " << numberOfMovesPerDepth[i]
              << " positions on depth: " << i + 1 << std::endl;
  }
  std::cout << "Best move found: " << MoveToUSI(bestMove) << std::endl;
  std::cout << "Time: " << duration.count() << " ms" << std::endl;
  return bestMove;
}

uint64_t countMovesGPU(Move* moves,
                       uint32_t size,
                       bool isWhite,
                       uint16_t depth,
                       uint16_t maxDepth,
                       GPUBuffer& gpuBuffer) {
  if (depth == maxDepth) {
    return size;
  }
  uint32_t processed = 0;
  uint32_t* offsets;
  uint32_t* bitboards;
  uint64_t numberOfMoves = 0;
  if (size == 0)
    size = 1;
  // Process by chunks
  while (processed < size) {
    uint32_t sizeToProcess = std::min(size - processed, maxProcessedSize);
    // Calculate offsets
    if (!gpuBuffer.ReserveOffsetsSpace(sizeToProcess + 1, offsets))
      printf("Err in ReserveOffsetsSpace\n");
    if (!gpuBuffer.ReserveBitboardsSpace(sizeToProcess, bitboards))
      printf("Err in ReserveBitboardsSpace\n");
    isWhite ? GPU::countWhiteMoves(sizeToProcess, depth,
                                   gpuBuffer.GetStartBoardPtr(), moves, size,
                                   processed, offsets, bitboards)
            : GPU::countBlackMoves(sizeToProcess, depth,
                                   gpuBuffer.GetStartBoardPtr(), moves, size,
                                   processed, offsets, bitboards);
    GPU::prefixSum(offsets, sizeToProcess + 1);
    uint32_t nextLayerSize = 0;
    cudaMemcpy(&nextLayerSize, offsets + sizeToProcess, sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    // Generate new moves
    Move* newMoves;

    if (!gpuBuffer.ReserveMovesSpace(nextLayerSize, depth + 1, newMoves))
      printf("Err in ReserveMovesSpace\n");
    isWhite ? GPU::generateWhiteMoves(sizeToProcess, depth,
                                      gpuBuffer.GetStartBoardPtr(), moves, size,
                                      processed, offsets, bitboards, newMoves)
            : GPU::generateBlackMoves(sizeToProcess, depth,
                                      gpuBuffer.GetStartBoardPtr(), moves, size,
                                      processed, offsets, bitboards, newMoves);
    gpuBuffer.FreeBitboardsSpace(sizeToProcess);
    // minmaxGPU(newLayer)
    numberOfMoves += countMovesGPU(newMoves, nextLayerSize, !isWhite, depth + 1,
                                   maxDepth, gpuBuffer);
    gpuBuffer.FreeMovesSpace(newMoves);
    gpuBuffer.FreeOffsetsSpace(offsets);
    processed += sizeToProcess;
  }
  return numberOfMoves;
}

Move GetBestMove(const Board& board,
                 bool isWhite,
                 uint16_t maxDepth,
                 uint32_t maxTime,
                 SearchType searchType) {
  terminateSearch.store(false);
  Move bestMove;
  std::future<void> future;
  if (searchType == CPU)
    future = std::async(IterativeDeepeningSearch, std::ref(bestMove), board,
                               isWhite, maxDepth, GetBestMoveCPU);
  else
    future = std::async(IterativeDeepeningSearch, std::ref(bestMove), board,
                               isWhite, maxDepth, GetBestMoveGPU);
  if (maxTime == 0)
    maxTime = UINT32_MAX;
  future.wait_for(std::chrono::milliseconds(maxTime));
  terminateSearch.store(true);
  future.wait();
  return bestMove;
}
}  // namespace SEARCH
}  // namespace engine
}  // namespace shogi