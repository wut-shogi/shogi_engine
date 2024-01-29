#include <iostream>
#include "multithreading.h"

namespace shogi {
namespace engine {
#ifdef __CUDACC__
std::mutex ThreadSafeLog::logMutex = std::mutex();
void ThreadSafeLog::WriteLine(const std::string& message) {
#ifdef  VERBOSE
  std::unique_lock<std::mutex> lock(logMutex);
  std::cout << message << std::endl;
#endif  //  VERBOSE
}

DevicePool::DevicePool(size_t numberOfDevices){
  for (size_t i = 0; i < numberOfDevices; ++i) {
    devicePool.push(i);
  }
}

int DevicePool::getDeviceIdFromPool() {
  std::unique_lock<std::mutex> lock(deviceMutex);
  condition.wait(lock, [this] { return !devicePool.empty(); });
  int deviceId = devicePool.top();
  devicePool.pop();
  return deviceId;
}

void DevicePool::releaseDeviceIdToPool(int deviceId) {
  {
    std::lock_guard<std::mutex> lock(deviceMutex);
    devicePool.push(deviceId);
  }
  condition.notify_one();
}
#endif
}  // namespace engine
}  // namespace shogi