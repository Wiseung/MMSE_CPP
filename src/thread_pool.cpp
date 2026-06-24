#include "internal/mmse_internal.h"

#include <condition_variable>
#include <mutex>
#include <thread>

namespace mmse::detail {

struct StaticThreadPool::Impl {
    std::mutex mutex;
    std::condition_variable cv_work;
    std::condition_variable cv_done;
    std::vector<std::thread> workers;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges;
    TaskFn fn = nullptr;
    void* ctx = nullptr;
    std::uint64_t generation = 0;
    std::uint64_t completed_generation = 0;
    std::uint32_t completed_workers = 0;
    std::uint32_t worker_count = 1;
    bool stop = false;
};

StaticThreadPool::StaticThreadPool() : impl_(std::make_unique<Impl>()) {}

StaticThreadPool::~StaticThreadPool() {
    stop();
}

void StaticThreadPool::init(std::uint32_t worker_count) {
    stop();

    impl_ = std::make_unique<Impl>();
    impl_->worker_count = worker_count == 0 ? 1U : worker_count;
    impl_->ranges.resize(impl_->worker_count);

    if (impl_->worker_count <= 1) {
        return;
    }

    for (std::uint32_t worker_id = 1; worker_id < impl_->worker_count; ++worker_id) {
        impl_->workers.emplace_back([this, worker_id]() {
            std::uint64_t observed_generation = 0;
            for (;;) {
                TaskFn fn = nullptr;
                void* ctx = nullptr;
                std::pair<std::uint32_t, std::uint32_t> range{};
                {
                    std::unique_lock lock(impl_->mutex);
                    impl_->cv_work.wait(lock, [&]() {
                        return impl_->stop || impl_->generation != observed_generation;
                    });
                    if (impl_->stop) {
                        return;
                    }
                    observed_generation = impl_->generation;
                    fn = impl_->fn;
                    ctx = impl_->ctx;
                    range = impl_->ranges[worker_id];
                }

                if (fn && range.first < range.second) {
                    fn(ctx, range.first, range.second);
                }

                {
                    std::lock_guard lock(impl_->mutex);
                    ++impl_->completed_workers;
                    if (impl_->completed_workers == impl_->worker_count - 1) {
                        impl_->completed_generation = observed_generation;
                        impl_->cv_done.notify_one();
                    }
                }
            }
        });
    }
}

void StaticThreadPool::stop() {
    if (!impl_) {
        return;
    }

    {
        std::lock_guard lock(impl_->mutex);
        impl_->stop = true;
        impl_->cv_work.notify_all();
    }

    for (std::thread& worker : impl_->workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    impl_->workers.clear();
}

std::uint32_t StaticThreadPool::worker_count() const {
    return impl_ ? impl_->worker_count : 1U;
}

void StaticThreadPool::parallel_for(
    const std::span<const std::pair<std::uint32_t, std::uint32_t>>& ranges,
    TaskFn fn,
    void* ctx) {
    if (!impl_ || impl_->worker_count <= 1 || ranges.size() <= 1) {
        if (!ranges.empty() && ranges[0].first < ranges[0].second) {
            fn(ctx, ranges[0].first, ranges[0].second);
        }
        return;
    }

    {
        std::lock_guard lock(impl_->mutex);
        for (std::size_t i = 0; i < ranges.size() && i < impl_->ranges.size(); ++i) {
            impl_->ranges[i] = ranges[i];
        }
        for (std::size_t i = ranges.size(); i < impl_->ranges.size(); ++i) {
            impl_->ranges[i] = {0U, 0U};
        }
        impl_->fn = fn;
        impl_->ctx = ctx;
        impl_->completed_workers = 0;
        ++impl_->generation;
        impl_->cv_work.notify_all();
    }

    if (ranges[0].first < ranges[0].second) {
        fn(ctx, ranges[0].first, ranges[0].second);
    }

    {
        std::unique_lock lock(impl_->mutex);
        impl_->cv_done.wait(lock, [&]() {
            return impl_->completed_generation == impl_->generation;
        });
    }
}

}  // namespace mmse::detail
