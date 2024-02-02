#pragma once
#include <mutex>
#include <condition_variable>
#include <vector>

template <typename T>
class ThreadSafeVector
{
public:
    void push_back(const T& value)
    {
        vector_.push_back(value);
        //lock.unlock();
        //condition_variable_.notify_one();
    }

    //void wait_for_elements(size_t count)
    //{
    //    std::unique_lock<std::mutex> lock(mutex_);
    //    condition_variable_.wait(lock, [this, count] { return vector_.size() >= count; });
    //}

    const std::vector<T>& get_vector() const
    {
        return vector_;
    }

private:
    std::vector<T> vector_{};
};