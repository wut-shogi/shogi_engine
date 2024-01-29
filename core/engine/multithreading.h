#include <future>
#include <mutex>
#include <stack>
#include <thread>
#include <string>

namespace shogi {
	namespace engine {
#ifdef __CUDACC__
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
  bool stop;
  std::stack<int> devicePool;
  int getDeviceIdFromPool();
  void releaseDeviceIdToPool(int deviceId);
};
#endif
}
}