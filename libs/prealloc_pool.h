#ifndef EMBEDLIC_PREALLOC_POOL_H
#define EMBEDLIC_PREALLOC_POOL_H

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <optional>
#include <mutex>

class PreAllocPool {
public:
    std::optional<std::pair<void*, void*>> acquire(size_t size) noexcept;
    void store(size_t size, std::pair<void*, void*> memory) noexcept;
    void release(size_t size, std::pair<void*, void*> memory);
private:
    struct hasher {
        inline std::size_t operator() (const std::pair<void*, void*> & v) const {
            return reinterpret_cast<size_t>(v.first) * 31 + reinterpret_cast<size_t>(v.second);
        }
    };

    std::unordered_map<std::size_t, std::unordered_set<std::pair<void*, void*>, hasher>> poolUnused;
    std::unordered_map<std::size_t, std::unordered_set<std::pair<void*, void*>, hasher>> poolUsed;

    std::mutex lock;
};

#endif //EMBEDLIC_PREALLOC_POOL_H
