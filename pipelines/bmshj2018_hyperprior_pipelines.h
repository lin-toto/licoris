#ifndef EMBEDLIC_BMSHJ2018_HYPERPRIOR_PIPELINES_H
#define EMBEDLIC_BMSHJ2018_HYPERPRIOR_PIPELINES_H

#include "libs/memory.h"
#include "libs/tensorrt_wrapper.h"
#include "entropy_coder/entropy_coder.h"
#include "libs/pipeline.h"
#include "pipelines/common_pipelines.h"

#include <chrono>
#include <string>
#include <utility>

// Compression
namespace Bmshj2018Hyperprior::Pipelines {
    struct CompressGpuTask : public TimedTask, GpuMemoryAwareTask<4, half> {
        size_t yBlockSize, zBlockSize, scalesBlockSize;
        unsigned int batchSize;

        CompressGpuTask(std::shared_ptr<MemoryWrapper<half>> xIn,
                        std::shared_ptr<MemoryWrapper<half>> yOut, std::shared_ptr<MemoryWrapper<half>> zOut, std::shared_ptr<MemoryWrapper<half>> scalesOut,
                        size_t yBlockSize, size_t zBlockSize, size_t scalesBlockSize, unsigned int batchSize)
                : TimedTask(),
                  GpuMemoryAwareTask(std::move(xIn), std::move(yOut), std::move(zOut), std::move(scalesOut)),
                  yBlockSize(yBlockSize), zBlockSize(zBlockSize), scalesBlockSize(scalesBlockSize), batchSize(batchSize) {}

        enum MemRefIndex {
            XIn = 0,
            YOut = 1,
            ZOut = 2,
            ScalesOut = 3
        };
    };

    struct CompressCpuTask : public TimedTask, GpuMemoryAwareTask<3, half> {
        size_t yOffset, zOffset, scalesOffset;

        CompressCpuTask(HighResolutionTimePoints timePoints,
                        std::shared_ptr<MemoryWrapper<half>> yOut, std::shared_ptr<MemoryWrapper<half>> zOut, std::shared_ptr<MemoryWrapper<half>> scalesOut,
                        size_t yOffset, size_t zOffset, size_t scalesOffset)
                : TimedTask(std::move(timePoints)),
                  GpuMemoryAwareTask(std::move(yOut), std::move(zOut), std::move(scalesOut)),
                  yOffset(yOffset), zOffset(zOffset), scalesOffset(scalesOffset) {}

        enum MemRefIndex {
            YOut = 0,
            ZOut = 1,
            ScalesOut = 2
        };
    };

    struct CompressFinalTask : public TimedTask {
        std::string yStrings, zStrings;
        CompressFinalTask(HighResolutionTimePoints timePoints, std::string&& yStrings, std::string&& zStrings)
                : TimedTask(std::move(timePoints)), yStrings(std::move(yStrings)), zStrings(std::move(zStrings)) {}
    };

    class CompressGpuPipeline : public GpuPipeline<4, CompressGpuTask, CompressCpuTask, half> {
    public:
        using GpuPipeline::GpuPipeline;
    private:
        std::vector<std::shared_ptr<CompressCpuTask>> work(std::shared_ptr<CompressGpuTask> gpuTask, unsigned int threadId) override;
    };

    class CompressCpuPipeline : public PipelineImpl<CompressCpuTask, CompressFinalTask> {
        using MyPipelineImpl = PipelineImpl<CompressCpuTask, CompressFinalTask>;
    public:
        CompressCpuPipeline(int width, int height, const std::string& bottleneckConstFile, const std::string& gaussianConstFile,
                            unsigned int maxTasks, std::shared_ptr<Pipeline> nextPipeline = nullptr)
                : MyPipelineImpl(maxTasks, std::move(nextPipeline)),
                  entropyBottleneck(128, ceil(height / 64), ceil(width / 64), std::make_shared<EntropyBottleneckConstantMap>(bottleneckConstFile)),
                  gaussianConditional(192, ceil(height / 16), ceil(width / 16), 0.1099, std::make_shared<GaussianConditionalConstantMap>(gaussianConstFile)) {}

    private:
        EntropyBottleneck entropyBottleneck;
        GaussianConditional gaussianConditional;

        std::vector<std::shared_ptr<CompressFinalTask>> work(std::shared_ptr<CompressCpuTask> cpuTask, unsigned int threadId) override;
    };

    class CompressStatisticsPipeline : public PipelineImpl<CompressFinalTask> {
        using MyPipelineImpl = PipelineImpl<CompressFinalTask>;
    public:
        CompressStatisticsPipeline(bool verbose, unsigned int maxTasks) : verbose(verbose), MyPipelineImpl(maxTasks, nullptr) {}

    private:
        bool verbose;
        unsigned int completed = 0;
        HighResolutionTime startTime;

        std::vector<std::shared_ptr<nullptr_t>> work(std::shared_ptr<CompressFinalTask> finalTask, unsigned int threadId) override;
    };
}

// Decompression
namespace Bmshj2018Hyperprior::Pipelines {
    struct DecompressGeneralTask : public TimedTask, GpuMemoryAwareTask<4, half> {
        std::string yStrings, zStrings;

        DecompressGeneralTask(std::string yStrings, std::string zStrings,
                              std::shared_ptr<MemoryWrapper<half>> zIn, std::shared_ptr<MemoryWrapper<half>> scalesOut,
                              std::shared_ptr<MemoryWrapper<half>> yIn, std::shared_ptr<MemoryWrapper<half>> xOut)
                : TimedTask(), yStrings(std::move(yStrings)), zStrings(std::move(zStrings)),
                  GpuMemoryAwareTask(std::move(zIn), std::move(scalesOut), std::move(yIn), std::move(xOut)) {}
        DecompressGeneralTask(HighResolutionTimePoints timePoints, std::string yStrings, std::string zStrings,
                              std::shared_ptr<MemoryWrapper<half>> zIn, std::shared_ptr<MemoryWrapper<half>> scalesOut,
                              std::shared_ptr<MemoryWrapper<half>> yIn, std::shared_ptr<MemoryWrapper<half>> xOut)
                : TimedTask(std::move(timePoints)), yStrings(std::move(yStrings)), zStrings(std::move(zStrings)),
                  GpuMemoryAwareTask(std::move(zIn), std::move(scalesOut), std::move(yIn), std::move(xOut)) {}

        enum MemRefIndex {
            ZIn = 0,
            ScalesOut = 1,
            YIn = 2,
            XOut = 3
        };
    };

    struct DecompressFinalTask : public TimedTask, GpuMemoryAwareTask<1, half> {
        DecompressFinalTask(HighResolutionTimePoints timePoints, std::shared_ptr<MemoryWrapper<half>> xOut)
                : TimedTask(std::move(timePoints)), GpuMemoryAwareTask(std::move(xOut)) {}

        enum MemRefIndex {
            XOut = 0
        };
    };

    class DecompressZHatPipeline : public PipelineImpl<DecompressGeneralTask, DecompressGeneralTask> {
        using MyPipelineImpl = PipelineImpl<DecompressGeneralTask, DecompressGeneralTask>;
    public:
        DecompressZHatPipeline(int width, int height, const std::string& bottleneckConstFile, unsigned int maxTasks,
                            std::shared_ptr<Pipeline> nextPipeline = nullptr)
                : MyPipelineImpl(maxTasks, std::move(nextPipeline)),
                  entropyBottleneck(128, ceil(height / 64), ceil(width / 64), std::make_shared<EntropyBottleneckConstantMap>(bottleneckConstFile)) {}

    private:
        EntropyBottleneck entropyBottleneck;

        std::vector<std::shared_ptr<DecompressGeneralTask>> work(std::shared_ptr<DecompressGeneralTask> generalTask, unsigned int threadId) override;
    };

    class DecompressScalesPipeline : public GpuPipeline<4, DecompressGeneralTask, DecompressGeneralTask, half> {
    public:
        using GpuPipeline::GpuPipeline;
    private:
        std::vector<std::shared_ptr<DecompressGeneralTask>> work(std::shared_ptr<DecompressGeneralTask> generalTask, unsigned int threadId) override;
    };

    class DecompressGaussianDecodePipeline : public PipelineImpl<DecompressGeneralTask, DecompressGeneralTask> {
        using MyPipelineImpl = PipelineImpl<DecompressGeneralTask, DecompressGeneralTask>;
    public:
        DecompressGaussianDecodePipeline(int width, int height, const std::string& gaussianConstFile, unsigned int maxTasks,
                                         std::shared_ptr<Pipeline> nextPipeline = nullptr)
                : MyPipelineImpl(maxTasks, std::move(nextPipeline)),
                  gaussianConditional(192, ceil(height / 16), ceil(width / 16), 0.1099, std::make_shared<GaussianConditionalConstantMap>(gaussianConstFile)) {}
    private:
        GaussianConditional gaussianConditional;

        std::vector<std::shared_ptr<DecompressGeneralTask>> work(std::shared_ptr<DecompressGeneralTask> generalTask, unsigned int threadId) override;
    };

    class DecompressYHatPipeline : public GpuPipeline<4, DecompressGeneralTask, DecompressFinalTask, half> {
    public:
        using GpuPipeline::GpuPipeline;
    private:

        std::vector<std::shared_ptr<DecompressFinalTask>> work(std::shared_ptr<DecompressGeneralTask> generalTask, unsigned int threadId) override;
    };

    class DecompressStatisticsPipeline : public PipelineImpl<DecompressFinalTask> {
        using MyPipelineImpl = PipelineImpl<DecompressFinalTask>;
    public:
        DecompressStatisticsPipeline(bool verbose, unsigned int maxTasks) : verbose(verbose), MyPipelineImpl(maxTasks, nullptr) {}

    private:
        bool verbose;
        unsigned int completed = 0;
        HighResolutionTime startTime;

        std::vector<std::shared_ptr<nullptr_t>> work(std::shared_ptr<DecompressFinalTask> task, unsigned int threadId) override;
    };
}

#endif //EMBEDLIC_BMSHJ2018_HYPERPRIOR_PIPELINES_H
