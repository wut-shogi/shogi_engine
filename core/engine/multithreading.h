#include <future>
#include <mutex>
#include <stack>
#include <string>
#include <thread>
#include <vector>

namespace shogi {
namespace engine {
#ifdef __CUDACC__
template <typename T>
class ThreadSafeVector {
 public:
  ThreadSafeVector(size_t size) { values = std::vector<T>(size, 0); }
  T GetValue(size_t idx) { return values[idx]; }

  void SetValue(size_t idx, T value) {
    std::lock_guard<std::mutex> lock(valuesMutex);
    values[idx] = value;
  }

  void AddValue(size_t idx, T value) {
    std::lock_guard<std::mutex> lock(valuesMutex);
    values[idx] += value;
  }

  size_t size() { return values.size(); }

 private:
  std::vector<T> values;
  std::mutex valuesMutex;
};

class ThreadSafeLog {
 public:
  static void WriteLine(const std::string& message);

 private:
  static std::mutex logMutex;
};

class DevicePool {
 public:
  DevicePool(size_t numberOfDevices);

  template <typename Result, typename Function, typename... Args>
  std::future<Result> executeWhenDeviceAvaliable(Function&& func,
                                                 Args&&... args) {
    return std::async(
        std::launch::async,
        [this, func = std::forward<Function>(func),
         argsTuple = std::make_tuple(std::forward<Args>(args)...)]() mutable {
          int deviceId = getDeviceIdFromPool();
          auto start = std::chrono::high_resolution_clock::now();
          ThreadSafeLog::WriteLine(
              "Starting thread with device Id: " + std::to_string(deviceId) +
              ", at: " + std::to_string(start.time_since_epoch().count()));
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
          ThreadSafeLog::WriteLine(
              "Ending thread with device Id: " + std::to_string(deviceId) +
              ", duration: " + std::to_string(duration.count()) + "");
          releaseDeviceIdToPool(deviceId);
          return result;
        });
  }

 private:
  std::mutex deviceMutex;
  std::condition_variable condition;
  std::stack<int> devicePool;
  int getDeviceIdFromPool();
  void releaseDeviceIdToPool(int deviceId);
};
#endif
}  // namespace engine
}  // namespace shogi