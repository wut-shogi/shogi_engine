#pragma once
#include <deque>
#include <mutex>

namespace shogi::engine::utils {
template <typename T>
class SafeQueue {
  std::deque<T> _queue;
  std::mutex _mutex;

 public:
  SafeQueue() = default;
  ~SafeQueue() = default;
  SafeQueue(SafeQueue<T>&) = delete;
  SafeQueue(SafeQueue<T>&& other) noexcept {
    std::lock_guard lock{other._mutex};
    this->_queue = std::move(other._queue);
  }

  SafeQueue<T>& operator=(SafeQueue<T> const&) = delete;
  SafeQueue<T>& operator=(SafeQueue<T>&&) = delete;

  void pushBack(T&& element);
  T&& popFront();
};

template <typename T>
void SafeQueue<T>::pushBack(T&& element) {
  std::lock_guard lock{_mutex};
  _queue.push_back(std::move(element));
}

template <typename T>
T&& SafeQueue<T>::popFront() {
  std::lock_guard lock{_mutex};
  T element = std::move(_queue.front());
  _queue.pop_front();
  return std::move(element);
}

}  // namespace shogi::engine::utils