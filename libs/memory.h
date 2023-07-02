#ifndef EMBEDLIC_MEMORY_H
#define EMBEDLIC_MEMORY_H

#ifndef NO_GPU_LIBRARIES
#include <cuda_runtime_api.h>
#endif

#include "prealloc_pool.h"
#include <cstdlib>
#include <array>
#include <type_traits>

#define ENABLE_POOLING

enum MemoryMode {
#ifndef NO_GPU_LIBRARIES
    NORMAL,
    PINNED,
    ZERO_COPY,
    UNIFIED,
#endif
    HOST_ONLY
};

enum Direction {
#ifndef NO_GPU_LIBRARIES
    HOST_TO_DEVICE,
    DEVICE_TO_HOST,
#endif
    UNSPECIFIED
};

extern std::array<PreAllocPool, static_cast<int>(HOST_ONLY) + 1> memPools;

template<typename T, typename = std::enable_if_t<std::is_trivial_v<T>>>
struct MemoryWrapper {
    friend class TensorRTWrapper;

    T *host;
    void *device;
    size_t size;
    MemoryMode memoryMode;
    Direction direction = UNSPECIFIED;

    MemoryWrapper(const MemoryWrapper&) = delete;
    MemoryWrapper& operator=(MemoryWrapper) = delete;

    MemoryWrapper(size_t size, MemoryMode memoryMode, Direction direction = UNSPECIFIED)
        : size(size), memoryMode(memoryMode), direction(direction) {
#ifdef ENABLE_POOLING
        auto ptr = memPools[memoryMode].acquire(size);
        if (ptr.has_value()) {
            host = static_cast<T*>(ptr->first);
            device = ptr->second;
        } else {
#endif
            switch (memoryMode) {
#ifndef NO_GPU_LIBRARIES
                case NORMAL:
                    host = static_cast<T*>(malloc(rawSize()));
                    cudaMalloc(&device, rawSize());
                    break;
                case PINNED:
                    cudaHostAlloc(reinterpret_cast<void**>(&host), rawSize(), cudaHostAllocDefault);
                    cudaMalloc(&device, rawSize());
                    break;
                case ZERO_COPY:
                    cudaHostAlloc(reinterpret_cast<void**>(&host), rawSize(), cudaHostAllocMapped);
                    cudaHostGetDevicePointer(&device, host, 0);
                    break;
                case UNIFIED:
                    cudaMallocManaged(&device, rawSize());
                    host = static_cast<T*>(device);
                    break;
#endif
                case HOST_ONLY:
                    host = static_cast<T*>(malloc(rawSize()));
                    device = nullptr;
                    break;
            }

#ifdef ENABLE_POOLING
            memPools[memoryMode].store(size, std::make_pair(host, device));
        }
#endif
    }

    ~MemoryWrapper() {
#ifdef ENABLE_POOLING
        memPools[memoryMode].release(size, std::make_pair(host, device));
#else
        switch (memoryMode) {
#ifndef NO_GPU_LIBRARIES
            case NORMAL:
                free(host);
                cudaFree(device);
                break;
            case PINNED:
                cudaFreeHost(host);
                cudaFree(device);
                break;
            case ZERO_COPY:
                cudaFreeHost(host);
                break;
            case UNIFIED:
                cudaFree(device);
                break;
#endif
            case HOST_ONLY:
                free(host);
                break;
        }
#endif
    }

    inline size_t rawSize() { return size * sizeof(T); }

#ifndef NO_GPU_LIBRARIES
    [[deprecated]]
    inline void copyToDevice () { doCopyToDevice(); }

    [[deprecated]]
    inline void copyToHost () { doCopyToHost(); }

private:
    inline void doCopyToDevice (cudaStream_t stream = nullptr, bool synchronized = true) {
        if (memoryMode == NORMAL) {
            cudaMemcpyAsync(device, host, rawSize(), cudaMemcpyHostToDevice, stream);
            if (synchronized) cudaStreamSynchronize(stream);
        }
    }

    inline void doCopyToHost (cudaStream_t stream = nullptr, bool synchronized = true) {
        if (memoryMode == NORMAL) {
            cudaMemcpyAsync(host, device, rawSize(), cudaMemcpyDeviceToHost, stream);
            if (synchronized) cudaStreamSynchronize(stream);
        }
    }
#endif
};


#endif //EMBEDLIC_MEMORY_H
