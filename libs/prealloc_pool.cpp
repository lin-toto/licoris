#include "prealloc_pool.h"

std::optional<std::pair<void *, void *>> PreAllocPool::acquire(size_t size) noexcept {
    const std::lock_guard<std::mutex> guard(lock);

    if (poolUnused.find(size) == poolUnused.end())return std::nullopt;
    if (poolUnused[size].empty()) return std::nullopt;

    auto it = poolUnused[size].begin();
    auto elem = *it;
    poolUnused[size].erase(it);

    if (poolUsed.find(size) == poolUsed.end()) poolUsed.emplace(size, std::initializer_list<std::pair<void*, void*>>{});
    poolUsed[size].emplace(elem);

    return elem;
}

void PreAllocPool::release(size_t size, std::pair<void *, void *> memory) {
    const std::lock_guard<std::mutex> guard(lock);

    if (poolUsed.find(size) == poolUsed.end()) throw std::runtime_error("Trying to release from non-existent size pool");

    auto it = poolUsed[size].find(memory);
    if (it == poolUsed[size].end()) throw std::runtime_error("Memory being released not in pool");

    auto elem = *it;
    poolUsed[size].erase(it);

    if (poolUnused.find(size) == poolUnused.end()) poolUnused.emplace(size, std::initializer_list<std::pair<void*, void*>>{});
    poolUnused[size].emplace(elem);
}

void PreAllocPool::store(size_t size, std::pair<void *, void *> memory) noexcept {
    const std::lock_guard<std::mutex> guard(lock);

    if (poolUsed.find(size) == poolUsed.end()) poolUsed.emplace(size, std::initializer_list<std::pair<void*, void*>>{});
    poolUsed[size].emplace(memory);
}