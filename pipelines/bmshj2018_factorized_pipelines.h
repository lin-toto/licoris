#ifndef EMBEDLIC_BMSHJ2018_FACTORIZED_PIPELINES_H
#define EMBEDLIC_BMSHJ2018_FACTORIZED_PIPELINES_H

#include "libs/memory.h"
#include "libs/tensorrt_wrapper.h"
#include "entropy_coder/entropy_coder.h"
#include "libs/pipeline.h"
#include "libs/network.h"
#include "pipelines/common_pipelines.h"
#include "libs/utils.h"
#include "libs/img_utils.h"

#include <opencv2/opencv.hpp>
#include <chrono>
#include <string>
#include <utility>

// Compression
namespace Bmshj2018Factorized::Pipelines {
    struct CompressGpuTask : public TimedTask, GpuMemoryAwareTask<2, half> {
        size_t outputBlockSize;
        unsigned int batchSize;

        CompressGpuTask(std::shared_ptr<MemoryWrapper<half>> gpuMemIn, std::shared_ptr<MemoryWrapper<half>> gpuMemOut,
                        size_t outputSize, unsigned int batchSize)
                : TimedTask(), GpuMemoryAwareTask(std::move(gpuMemIn), std::move(gpuMemOut)),
                  outputBlockSize(outputSize), batchSize(batchSize) {}

        enum MemRefIndex {
            In = 0, Out = 1
        };
    };

    struct CompressCpuTask : public TimedTask, GpuMemoryAwareTask<1, half> {
        size_t inputOffset;

        CompressCpuTask(HighResolutionTimePoints timePoints,
                        std::shared_ptr<MemoryWrapper<half>> gpuMemOut, size_t inputOffset)
                : TimedTask(std::move(timePoints)), GpuMemoryAwareTask(std::move(gpuMemOut)),
                  inputOffset(inputOffset) {}

        enum MemRefIndex {
            Out = 0
        };
    };

    struct CompressFinalTask : public TimedTask {
        std::string result;

        CompressFinalTask(HighResolutionTimePoints timePoints, std::string&& result)
                : TimedTask(std::move(timePoints)), result(std::move(result)) {}
    };

    class CompressGpuPipeline : public GpuPipeline<2, CompressGpuTask, CompressCpuTask, half> {
    public:
        using GpuPipeline::GpuPipeline;
    private:
        std::vector<std::shared_ptr<CompressCpuTask>> work(std::shared_ptr<CompressGpuTask> gpuTask, unsigned int threadId) override;
    };

    class CompressCpuPipeline : public PipelineImpl<CompressCpuTask, CompressFinalTask> {
        using MyPipelineImpl = PipelineImpl<CompressCpuTask, CompressFinalTask>;
    public:
        CompressCpuPipeline(int width, int height, const std::string& coderConstFile, unsigned int maxTasks,
                            std::shared_ptr<Pipeline> nextPipeline = nullptr)
                : MyPipelineImpl(maxTasks, std::move(nextPipeline)),
                  coder(192, ceil(height / 16), ceil(width / 16), std::make_shared<EntropyBottleneckConstantMap>(coderConstFile)) {}

    private:
        EntropyBottleneck coder;

        std::vector<std::shared_ptr<CompressFinalTask>> work(std::shared_ptr<CompressCpuTask> cpuTask, unsigned int threadId) override;
    };

    class CompressStatisticsPipeline : public PipelineImpl<CompressFinalTask> {
        using MyPipelineImpl = PipelineImpl<CompressFinalTask>;
    public:
        CompressStatisticsPipeline(bool verbose, unsigned int maxTasks)
            : verbose(verbose), MyPipelineImpl(maxTasks, nullptr) {}

    private:
        bool verbose;
        unsigned int completed = 0;
        HighResolutionTime startTime;

        std::vector<std::shared_ptr<nullptr_t>> work(std::shared_ptr<CompressFinalTask> finalTask, unsigned int threadId) override;
    };

    class CompressTCPSenderPipeline : public PipelineImpl<CompressFinalTask> {
        using MyPipelineImpl = PipelineImpl<CompressFinalTask>;
    public:
        CompressTCPSenderPipeline(bool verbose, TCPSender &sender, unsigned int maxTasks)
                : verbose(verbose), sender(sender), MyPipelineImpl(maxTasks, nullptr) {}

    private:
        bool verbose;
        TCPSender &sender;

        std::vector<std::shared_ptr<nullptr_t>> work(std::shared_ptr<CompressFinalTask> finalTask, unsigned int threadId) override;
    };
}

// Decompression
namespace Bmshj2018Factorized::Pipelines {
    struct DecompressCpuTask : public TimedTask, GpuMemoryAwareTask<2, half> {
        std::string input;

        DecompressCpuTask(std::shared_ptr<MemoryWrapper<half>> gpuMemIn, std::shared_ptr<MemoryWrapper<half>> gpuMemOut,
                          std::string input)
                : TimedTask(), GpuMemoryAwareTask(std::move(gpuMemIn), std::move(gpuMemOut)),
                  input(std::move(input)) {}

        enum MemRefIndex {
            In = 0, Out = 1
        };
    };

    struct DecompressGpuTask : public TimedTask, GpuMemoryAwareTask<2, half> {
        DecompressGpuTask(HighResolutionTimePoints timePoints,
                          std::shared_ptr<MemoryWrapper<half>> gpuMemIn, std::shared_ptr<MemoryWrapper<half>> gpuMemOut)
                : TimedTask(std::move(timePoints)), GpuMemoryAwareTask(std::move(gpuMemIn), std::move(gpuMemOut)) {}

        enum MemRefIndex {
            In = 0, Out = 1
        };
    };

    struct DecompressFinalTask : public TimedTask, GpuMemoryAwareTask<1, half> {
        DecompressFinalTask(HighResolutionTimePoints timePoints, std::shared_ptr<MemoryWrapper<half>> gpuMemOut)
                : TimedTask(std::move(timePoints)), GpuMemoryAwareTask(std::move(gpuMemOut)) {}

        enum MemRefIndex {
            Out = 0
        };
    };

    class DecompressCpuPipeline : public PipelineImpl<DecompressCpuTask, DecompressGpuTask> {
        using MyPipelineImpl = PipelineImpl<DecompressCpuTask, DecompressGpuTask>;
    public:
        DecompressCpuPipeline(int width, int height, const std::string& coderConstFile, unsigned int maxTasks,
                              std::shared_ptr<Pipeline> nextPipeline = nullptr)
                : MyPipelineImpl(maxTasks, std::move(nextPipeline)),
                coder(192, ceil(height / 16), ceil(width / 16), std::make_shared<EntropyBottleneckConstantMap>(coderConstFile)) {}

    private:
        EntropyBottleneck coder;

        std::vector<std::shared_ptr<DecompressGpuTask>> work(std::shared_ptr<DecompressCpuTask> cpuTask, unsigned int threadId) override;
    };

    class DecompressGpuPipeline : public GpuPipeline<2, DecompressGpuTask, DecompressFinalTask, half> {
    public:
        using GpuPipeline::GpuPipeline;
    private:
        std::vector<std::shared_ptr<DecompressFinalTask>> work(std::shared_ptr<DecompressGpuTask> gpuTask, unsigned int threadId) override;
    };

    class DecompressStatisticsPipeline : public PipelineImpl<DecompressFinalTask> {
        using MyPipelineImpl = PipelineImpl<DecompressFinalTask>;
    public:
        DecompressStatisticsPipeline(bool verbose, unsigned int maxTasks) : verbose(verbose), MyPipelineImpl(maxTasks, nullptr) {}

    private:
        bool verbose;
        unsigned int completed = 0;
        HighResolutionTime startTime;

        std::vector<std::shared_ptr<nullptr_t>> work(std::shared_ptr<DecompressFinalTask> finalTask, unsigned int threadId) override;
    };

    class DecompressWindowPipeline : public PipelineImpl<DecompressFinalTask> {
        using MyPipelineImpl = PipelineImpl<DecompressFinalTask>;
    public:
        DecompressWindowPipeline(int width, int height, bool verbose, std::string imagePath, unsigned int maxTasks)
            : MyPipelineImpl(maxTasks, nullptr), width(width), height(height), verbose(verbose), imagePath(std::move(imagePath)) {
            cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        }

        std::vector<std::shared_ptr<nullptr_t>> work(std::shared_ptr<DecompressFinalTask> finalTask, unsigned int threadId) override;
    private:
        int width, height;
        bool verbose;
        std::string imagePath;
        const std::string windowName = "Decompress Demo";
    };

}

#endif //EMBEDLIC_BMSHJ2018_FACTORIZED_PIPELINES_H
