#pragma once
#include <deque>
#include <mutex>

namespace shogi {
namespace engine {
namespace utils {
template <typename T>
class thread_safe_queue {
  std::deque<T> _queue;
  std::mutex _mutex;

 public:
  thread_safe_queue(){};
  thread_safe_queue(thread_safe_queue<T>&) = delete;

  thread_safe_queue(thread_safe_queue<T>&& other) {
    this->_queue = std::move(other._queue);
    this->_mutex = std::move(other._mutex);
  }

  thread_safe_queue<T>& operator=(thread_safe_queue<T>&& other) {
    if (&other != this) {
      *this = std::move(other);
    }
    return *this;
  };

  void push_back(T&& element);
  T&& pop_front();
};

template <typename T>
void thread_safe_queue<T>::push_back(T&& element) {
  std::lock_guard lock{_mutex};
  _queue.push_back(std::move(element));
}

template <typename T>
T&& thread_safe_queue<T>::pop_front() {
  std::lock_guard lock{_mutex};
  T element = std::move(_queue.front());
  _queue.pop_front();
  return std::move(element);
}

}  // namespace utils
}  // namespace engine
}  // namespace shogi