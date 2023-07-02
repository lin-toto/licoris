#ifndef EMBEDLIC_COMMON_PIPELINES_H
#define EMBEDLIC_COMMON_PIPELINES_H

#include "libs/pipeline.h"

#ifndef NO_GPU_LIBRARIES
#include "libs/tensorrt_wrapper.h"
#endif

#include <chrono>
#include <unordered_map>

using HighResolutionTime = std::chrono::time_point<std::chrono::high_resolution_clock>;
using HighResolutionTimePoints = std::unordered_map<std::string, HighResolutionTime>;

struct TimedTask : public virtual ITask {
    HighResolutionTimePoints timePoints;
    HighResolutionTime timer;

    TimedTask() { recordTimePoint("construct"); }
    explicit TimedTask(HighResolutionTimePoints timePoints) : timePoints(std::move(timePoints)) {}

    inline void recordTimePoint(const std::string& key) {
        timePoints[key] = std::chrono::high_resolution_clock::now();
    }

    inline unsigned int consumedTime(const std::string& startKey, const std::string& endKey) {
        return std::chrono::duration_cast<std::chrono::microseconds>(timePoints[startKey] - timePoints[endKey]).count();
    }
};

#ifndef NO_GPU_LIBRARIES
template <unsigned long count, typename T>
struct GpuMemoryAwareTask : public virtual ITask {
    std::array<std::shared_ptr<MemoryWrapper<T>>, count> memRefs;

    template<typename ...Args,
            typename = std::enable_if_t<std::conjunction_v<std::is_same<std::shared_ptr<MemoryWrapper<T>>, Args>...>>>
    explicit GpuMemoryAwareTask(Args ... ref): memRefs({std::move(ref) ...}) {}
};

template <unsigned long count, typename AcceptedTask, typename ProducedTask, typename MemType,
        typename = std::enable_if_t<std::is_base_of_v<GpuMemoryAwareTask<count, MemType>, AcceptedTask>>>
class GpuPipeline : public PipelineImpl<AcceptedTask, ProducedTask> {
    using MyPipelineImpl = PipelineImpl<AcceptedTask, ProducedTask>;
public:
    GpuPipeline(TensorRTWrapper& trt, unsigned int maxTasks,
            std::shared_ptr<Pipeline> nextPipeline = nullptr)
    : trt(trt), MyPipelineImpl(maxTasks, std::move(nextPipeline)) {}

protected:
    void workGpu(const std::shared_ptr<GpuMemoryAwareTask<count, MemType>>& gpuTask) {
        for (auto& memRef: gpuTask->memRefs) {
            if (memRef->direction == HOST_TO_DEVICE)
                trt.copyToDevice(memRef);
        }
        trt.execute<count, MemType>(gpuTask->memRefs);
        for (auto& memRef: gpuTask->memRefs) {
            if (memRef->direction == DEVICE_TO_HOST)
                trt.copyToHost(memRef);
        }
        trt.sync();
    }

    TensorRTWrapper& trt;
};
#endif

#endif //EMBEDLIC_COMMON_PIPELINES_H
