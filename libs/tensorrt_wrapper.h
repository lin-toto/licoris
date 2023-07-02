#ifndef EMBEDLIC_TENSORRT_WRAPPER_H
#define EMBEDLIC_TENSORRT_WRAPPER_H

#include <exception>
#include <iostream>
#include <iterator>
#include <fstream>
#include <string>
#include <memory>
#include <vector>
#include <initializer_list>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <NvInfer.h>

#include "memory.h"

template <typename T>
struct accept_memory_wrapper { static constexpr bool value = false; };

template <typename T>
struct accept_memory_wrapper<MemoryWrapper<T>> { static constexpr bool value = true; };

class TensorRTWrapper {
public:
    TensorRTWrapper(const std::string& modelPath, nvinfer1::ILogger& logger, bool synchronized = true);
    ~TensorRTWrapper();
    TensorRTWrapper(const TensorRTWrapper&) = delete;
    TensorRTWrapper& operator=(TensorRTWrapper) = delete;

    void setInputDimension(int index, nvinfer1::Dims dims);
    nvinfer1::Dims getOutputDimension(int index);

    template<class T>
    inline void copyToHost(const std::shared_ptr<MemoryWrapper<T>>& mem) {
        mem->doCopyToHost(stream, synchronized);
    }

    template<class T>
    inline void copyToDevice(const std::shared_ptr<MemoryWrapper<T>>& mem) {
        mem->doCopyToDevice(stream, synchronized);
    }

    inline void setDLACore(int id) { runtime->setDLACore(id); }

    inline void sync() { if (!synchronized) cudaStreamSynchronize(stream); }

    template<typename ...Args,
            typename = std::enable_if_t<std::conjunction_v<accept_memory_wrapper<Args>...>>>
    inline void execute(const std::shared_ptr<Args>& ... list) {
        void *bindings[sizeof...(list)] = {list->device...};
        doExecute(bindings);
    }

    template<unsigned long count, typename T>
    inline void execute(const std::array<std::shared_ptr<MemoryWrapper<T>>, count>& list) {
        void **bindings = new void*[count];
        for (unsigned int i = 0; i < count; i++) {
            bindings[i] = list[i]->device;
        }
        doExecute(bindings);
        delete[] bindings;
    }

    template<typename ...Args,
            typename = std::enable_if_t<std::conjunction_v<accept_memory_wrapper<Args>...>>>
    inline void warmup(int rounds, const std::shared_ptr<Args>& ... list) {
        for (int i = 0; i < rounds; i++) execute(list...);
    }
private:
    nvinfer1::ILogger& logger;
    cudaStream_t stream{ nullptr };

    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    bool synchronized;

    void doExecute(void **bindings);
};


#endif //EMBEDLIC_TENSORRT_WRAPPER_H
