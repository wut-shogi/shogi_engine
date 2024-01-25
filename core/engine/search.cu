#include <atomic>
#include <future>
#include <mutex>
#include <queue>
#include <stack>
#include <thread>
#include <vector>
#include "CPUsearchHelpers.h"
#include "GPUsearchHelpers.h"
#include "USIconverter.h"
#include "evaluation.h"
#include "lookUpTables.h"
#include "search.h"

#ifdef __CUDACC__
#include <thrust/extrema.h>
#endif

namespace shogi {
namespace engine {
namespace SEARCH {

#ifdef __CUDACC__
class ThreadSafeLog {
 public:
  void WriteLine(const std::string& message) {
    std::unique_lock<std::mutex> lock(logMutex);
    std::cout << message << std::endl;
  }

 private:
  std::mutex logMutex;
};
class DevicePool {
 public:
  DevicePool(size_t numberOfDevices) : stop(false) {
    for (size_t i = 0; i < numberOfDevices; ++i) {
      devicePool.push(i);
    }
  }

  template <typename Result, typename Function, typename... Args>
  std::future<Result> executeWhenDeviceAvaliable(Function&& func,
                                                 Args&&... args) {
    return std::async(
        std::launch::async,
        [this, func = std::forward<Function>(func),
         argsTuple = std::make_tuple(std::forward<Args>(args)...)]() mutable {
          int deviceId = getDeviceIdFromPool();
          auto start = std::chrono::high_resolution_clock::now();
          logger.WriteLine("Starting thread with device Id: " +
                           std::to_string(deviceId) + ", at: " + std::to_string(start.time_since_epoch().count()));
          cudaSetDevice(deviceId);
          Result result = std::apply(
              [func, deviceId](auto&&... funcArgs) mutable {
                return func(deviceId,
                            std::forward<decltype(funcArgs)>(funcArgs)...);
              },
              argsTuple);
          auto stop = std::chrono::high_resolution_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
              stop - start);
          logger.WriteLine(
              "Ending thread with device Id: " + std::to_string(deviceId) +
              ", duration: " + std::to_string(duration.count()) + "");
          releaseDeviceIdToPool(deviceId);
          return result;
        });
  }

 private:
  std::mutex deviceMutex;
  std::condition_variable condition;
  bool stop;
  ThreadSafeLog logger;

  std::stack<int> devicePool;

  int getDeviceIdFromPool() {
    std::unique_lock<std::mutex> lock(deviceMutex);
    // Wait until a device Id is available
    condition.wait(lock, [this] { return !devicePool.empty(); });
    // Acquire an available device Id
    int deviceId = devicePool.top();
    devicePool.pop();
    return deviceId;
  }

  void releaseDeviceIdToPool(int deviceId) {
    std::lock_guard<std::mutex> lock(deviceMutex);
    devicePool.push(deviceId);
    condition.notify_one();
  }
};
#endif

struct DeviceData {
  uint8_t* buffer = nullptr;
  uint32_t bufferSize = 0;
};

std::atomic<bool> terminateSearch(false);

bool afterInit = false;

int numberOfDevices = 1;

std::vector<DeviceData> deviceData(numberOfDevices);

bool initDevice(int deviceId) {
#ifdef __CUDACC__
  LookUpTables::GPU::init();
  size_t total = 0, free = 0;
  cudaMemGetInfo(&free, &total);
  if (free == 0)
    return false;
  deviceData[deviceId].bufferSize = (free / 4) * 4;
  cudaError_t error = cudaMalloc((void**)&(deviceData[deviceId].buffer),
                                 deviceData[deviceId].bufferSize);
  if (error != cudaSuccess)
    return false;
#endif
  return true;
}

bool init() {
  bool finalResult = true;
  try {
    LookUpTables::CPU::init();
#ifdef __CUDACC__
    bool finalResult = true;
    DevicePool devicePool(numberOfDevices);
    std::vector<std::future<bool>> futures;
    for (int device = 0; device < numberOfDevices; device++) {
      futures.emplace_back(
          devicePool.executeWhenDeviceAvaliable<bool>(initDevice));
    }
    for (auto& future : futures) {
      future.wait();
      bool result = future.get();
      finalResult &= result;
    }
#endif
  } catch (...) {
    return false;
  }
  afterInit = finalResult;
  return finalResult;
}

bool cleanupDevice(int deviceId) {
#ifdef __CUDACC__
  LookUpTables::GPU::cleanup();
  if (deviceData[deviceId].bufferSize > 0)
    cudaFree(deviceData[deviceId].buffer);
#endif
  return true;
}

void cleanup() {
  LookUpTables::CPU::cleanup();
#ifdef __CUDACC__
  DevicePool devicePool(numberOfDevices);
  std::vector<std::future<bool>> futures;
  for (int device = 0; device < numberOfDevices; device++) {
    futures.emplace_back(
        devicePool.executeWhenDeviceAvaliable<bool>(cleanupDevice));
  }
  for (auto& future : futures) {
    future.wait();
  }
#endif
  afterInit = false;
}

void setDeviceCount(int numberOfDevicesUsed) {
#ifdef __CUDACC__
  if (afterInit) {
    printf("Cannot change number of devices after initialization\n");
  }
  int numberOfAllDevices = 1;
  cudaGetDeviceCount(&numberOfAllDevices);
  numberOfDevices = std::min(numberOfAllDevices, numberOfDevicesUsed);
  deviceData.resize(numberOfDevices);
  if (numberOfDevices != numberOfDevicesUsed) {
    printf("Cannot use %d devices. Using %d devices instead\n",
           numberOfDevicesUsed, numberOfDevices);
  } else {
    printf("Using %d GPUs out of %d avaliable\n", numberOfDevices,
           numberOfAllDevices);
  }
#endif  //  __CUDACC__
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
// #define __CUDACC__
#ifdef __CUDACC__
template <typename T>
class ThreadSafeVector {
 public:
  ThreadSafeVector(size_t size) { values = std::vector<T>(size, 0); }
  T& operator[](int idx) {
    std::lock_guard<std::mutex> lock(valuesMutex);
    return values[idx];
  }

  size_t size() { return values.size(); }

 private:
  std::vector<T> values;
  std::mutex valuesMutex;
};

GPUBuffer::GPUBuffer(const Board& startBoard,
                     uint8_t* d_buffer,
                     uint32_t size) {
  buffer = d_buffer;
  bufferSize = size;
  d_startBoard = (Board*)buffer;
  cudaMemcpy(d_startBoard, &startBoard, sizeof(Board), cudaMemcpyHostToDevice);
  freeBegin = buffer + sizeof(Board);
  freeEnd = buffer + bufferSize;
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

void GPUBuffer::FreeBitboardsSpace() {
  freeEnd = buffer + bufferSize;
}

static const uint32_t maxProcessedSize = 50000;

// Converts moves to values
int minMaxGPU(Move* moves,
              uint32_t size,
              bool isWhite,
              uint16_t depth,
              uint16_t maxDepth,
              GPUBuffer& gpuBuffer,
              ThreadSafeVector<uint64_t>& numberOfMovesPerDepth) {
  if (terminateSearch)
    return 0;
  if (depth == maxDepth) {
    // Evaluate moves
    GPU::evaluateBoards(size, isWhite, depth, gpuBuffer.GetStartBoardPtr(),
                        moves, (int16_t*)moves);
    numberOfMovesPerDepth[depth - 1] += size;
    return 0;
  }
  uint32_t processed = 0;
  uint32_t* offsets;
  uint32_t* bitboards;
  uint32_t* bestIndex;

  int result = 0;
  // Process by chunks
  while (processed < size) {
    uint32_t sizeToProcess = std::min(size - processed, maxProcessedSize);
    // Calculate offsets
    if (!gpuBuffer.ReserveOffsetsSpace(sizeToProcess + 1, offsets))
      printf("Err in ReserveOffsetsSpace\n");
    if (!gpuBuffer.ReserveBitboardsSpace(sizeToProcess, bitboards))
      printf("Err in ReserveBitboardsSpace\n");
    result = isWhite ? GPU::countWhiteMoves(sizeToProcess, depth,
                                            gpuBuffer.GetStartBoardPtr(), moves,
                                            size, processed, offsets, bitboards)
                     : GPU::countBlackMoves(
                           sizeToProcess, depth, gpuBuffer.GetStartBoardPtr(),
                           moves, size, processed, offsets, bitboards);
    if (result)
      return result;
    result = GPU::prefixSum(offsets, sizeToProcess + 1);
    if (result)
      return result;
    uint32_t nextLayerSize = 0;
    cudaMemcpy(&nextLayerSize, offsets + sizeToProcess, sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    // Generate new moves
    Move* newMoves;

    if (!gpuBuffer.ReserveMovesSpace(nextLayerSize, depth + 1, newMoves))
      printf("Err in ReserveMovesSpace\n");
    result = isWhite
                 ? GPU::generateWhiteMoves(
                       sizeToProcess, depth, gpuBuffer.GetStartBoardPtr(),
                       moves, size, processed, offsets, bitboards, newMoves)
                 : GPU::generateBlackMoves(
                       sizeToProcess, depth, gpuBuffer.GetStartBoardPtr(),
                       moves, size, processed, offsets, bitboards, newMoves);
    if (result)
      return result;
    gpuBuffer.FreeBitboardsSpace();
    // minmaxGPU(newLayer)
    result = minMaxGPU(newMoves, nextLayerSize, !isWhite, depth + 1, maxDepth,
                       gpuBuffer, numberOfMovesPerDepth);
    if (result)
      return result;
    bestIndex = offsets;
    // Gather values from new layer
    result = isWhite ? GPU::gatherValuesMax(
                           sizeToProcess, depth, offsets, (int16_t*)newMoves,
                           (int16_t*)(moves + processed), bestIndex)
                     : GPU::gatherValuesMin(
                           sizeToProcess, depth, offsets, (int16_t*)newMoves,
                           (int16_t*)(moves + processed), bestIndex);
    if (result)
      return result;
    gpuBuffer.FreeMovesSpace(newMoves);
    gpuBuffer.FreeOffsetsSpace(offsets);
    processed += sizeToProcess;
  }
  numberOfMovesPerDepth[depth - 1] += size;
  return 0;
}

int findMoveValueDevice(int deviceId,
                        uint32_t size,
                        Move* inMoves,
                        int16_t* outValues,
                        const Board& board,
                        bool isWhite,
                        uint16_t maxDepth,
                        ThreadSafeVector<uint64_t>& numberOfMovesPerDepth) {
  cudaSetDevice(deviceId);
  GPUBuffer gpuBuffer(board, deviceData[deviceId].buffer,
                      deviceData[deviceId].bufferSize);
  Move* d_moves;
  gpuBuffer.ReserveMovesSpace(size, 1, d_moves);
  cudaMemcpy(d_moves, inMoves, size * sizeof(Move), cudaMemcpyHostToDevice);
  int result = minMaxGPU(d_moves, size, !isWhite, 1, maxDepth, gpuBuffer,
                         numberOfMovesPerDepth);
  if (result)
    return result;
  cudaMemcpy(outValues, d_moves, size * sizeof(int16_t),
             cudaMemcpyDeviceToHost);
  return 0;
}

Move GetBestMoveGPU(const Board& board, bool isWhite, uint16_t maxDepth) {
  auto start = std::chrono::high_resolution_clock::now();
  ThreadSafeVector<uint64_t> numberOfMovesPerDepth(maxDepth);
  CPU::MoveList rootMoves(board, isWhite);
  if (rootMoves.size() == 0) {
    return Move{0, 0, 0};
  }
  Move bestMove = *rootMoves.begin();
  std::vector<int16_t> h_values(rootMoves.size());
  uint32_t avgSize = rootMoves.size() / numberOfDevices;
  uint32_t processedMoves = 0;
  std::vector<std::future<int>> futures;
  for (int device = 0; device < numberOfDevices; device++) {
    uint32_t size = std::min(avgSize, rootMoves.size() - processedMoves);
    futures.emplace_back(std::async(
        std::launch::async, findMoveValueDevice, device, size,
        rootMoves.data() + processedMoves, h_values.data() + processedMoves,
        std::cref(board), isWhite, maxDepth, std::ref(numberOfMovesPerDepth)));
    processedMoves += size;
  }
  for (auto& future : futures) {
    future.wait();
    int result = future.get();
    if (result)
      return Move{0, 0, 0};
  }
  if (terminateSearch)
    return bestMove;

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

uint64_t countMovesDevice(Move* moves,
                          uint32_t size,
                          bool isWhite,
                          uint16_t depth,
                          uint16_t maxDepth,
                          GPUBuffer& gpuBuffer) {
  if (depth == maxDepth)
    return size;
  uint32_t processed = 0;
  uint32_t* offsets;
  uint32_t* bitboards;
  uint64_t numberOfMoves = 0;
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
    gpuBuffer.FreeBitboardsSpace();

    numberOfMoves += countMovesDevice(newMoves, nextLayerSize, !isWhite,
                                      depth + 1, maxDepth, gpuBuffer);
    gpuBuffer.FreeMovesSpace(newMoves);
    gpuBuffer.FreeOffsetsSpace(offsets);
    processed += sizeToProcess;
  }
  return numberOfMoves;
}

uint64_t launchCountMovesDevice(int deviceId,
                                Board& board,
                                Move move,
                                bool isWhite,
                                uint16_t maxDepth) {
  GPUBuffer gpuBuffer(board, deviceData[deviceId].buffer,
                      deviceData[deviceId].bufferSize);
  Move* d_moves;
  gpuBuffer.ReserveMovesSpace(1, 1, d_moves);
  cudaMemcpy(d_moves, &move, sizeof(Move), cudaMemcpyHostToDevice);
  uint64_t numberOfMoves =
      countMovesDevice(d_moves, 1, isWhite, 1, maxDepth, gpuBuffer);
  return numberOfMoves;
}

uint64_t countMovesGPU(bool Verbose,
                       const Board& board,
                       CPU::MoveList& moves,
                       bool isWhite,
                       uint16_t maxDepth) {
  DevicePool devicePool(numberOfDevices);
  std::vector<std::pair<int, std::future<uint64_t>>> futures;
  uint64_t nodesSearched = 0;
  for (int i = 0; i < moves.size(); i++) {
    futures.emplace_back(i, devicePool.executeWhenDeviceAvaliable<uint64_t>(
                                launchCountMovesDevice, board,
                                *(moves.data() + i), !isWhite, maxDepth));
  }
  while (!futures.empty()) {
    auto it = futures.begin();
    while (it != futures.end()) {
      auto status = it->second.wait_for(std::chrono::milliseconds(0));
      if (status == std::future_status::ready) {
        uint64_t numberOfMoves = it->second.get();
        int moveIdx = it->first;
        if (Verbose)
          std::cout << MoveToUSI(*(moves.data() + moveIdx)) << ": "
                    << numberOfMoves << std::endl;
        nodesSearched += numberOfMoves;
        it = futures.erase(it);
      } else {
        ++it;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return nodesSearched;
}

#endif

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
  else {
#ifdef __CUDACC__
    future = std::async(IterativeDeepeningSearch, std::ref(bestMove), board,
                        isWhite, maxDepth, GetBestMoveGPU);
#else
    future = std::async(IterativeDeepeningSearch, std::ref(bestMove), board,
                        isWhite, maxDepth, GetBestMoveCPU);
#endif
  }
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