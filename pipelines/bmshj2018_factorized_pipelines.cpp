#include "bmshj2018_factorized_pipelines.h"

using namespace Bmshj2018Factorized::Pipelines;

std::vector<std::shared_ptr<CompressCpuTask>> CompressGpuPipeline::work(std::shared_ptr<CompressGpuTask> gpuTask, unsigned int threadId) {
    gpuTask->recordTimePoint("gpu_start");
    workGpu(gpuTask);
    gpuTask->recordTimePoint("gpu_end");

    std::vector<std::shared_ptr<CompressCpuTask>> result;
    result.reserve(gpuTask->batchSize);
    for (unsigned int i = 0; i < gpuTask->batchSize; i++) {
        size_t cpuInputOffset = i * gpuTask->outputBlockSize;
        result.push_back(
                std::make_shared<CompressCpuTask>(gpuTask->timePoints, gpuTask->memRefs[CompressGpuTask::Out], cpuInputOffset)
        );
    }

    return result;
}

std::vector<std::shared_ptr<CompressFinalTask>> CompressCpuPipeline::work(std::shared_ptr<CompressCpuTask> cpuTask, unsigned int threadId) {
    cpuTask->recordTimePoint("cpu_start");

    auto *input = cpuTask->memRefs[CompressCpuTask::Out]->host + cpuTask->inputOffset;
    std::string result = coder.compress(input);

    cpuTask->recordTimePoint("cpu_end");

    return {std::make_shared<CompressFinalTask>(std::move(cpuTask->timePoints), std::move(result))};
}

std::vector<std::shared_ptr<nullptr_t>> CompressStatisticsPipeline::work(std::shared_ptr<CompressFinalTask> finalTask, unsigned int threadId) {
    if (verbose) {
        auto totalElapsed = finalTask->consumedTime("cpu_end", "gpu_start");
        auto gpuElapsed = finalTask->consumedTime("gpu_end", "gpu_start");
        auto cpuElapsed = finalTask->consumedTime("cpu_end", "cpu_start");
        auto others = totalElapsed - gpuElapsed - cpuElapsed;
        std::cout << "Elapsed: " << totalElapsed << "us, GPU: " << gpuElapsed << "us, CPU: " << cpuElapsed
                  << "us, Others: " << others << "us" << std::endl;
    }

    completed++;
    if (completed == 500) {
        startTime = std::chrono::high_resolution_clock::now();
    } else if (completed % 500 == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - startTime).count();
        std::cout << "Completed: " << completed << ", FPS: " << (completed - 500) / duration << std::endl;
    }

    return {};
}

std::vector<std::shared_ptr<nullptr_t>> CompressTCPSenderPipeline::work(std::shared_ptr<CompressFinalTask> finalTask, unsigned int threadId) {
    auto frameLatency = finalTask->consumedTime("cpu_end", "construct");
    auto totalElapsed = finalTask->consumedTime("cpu_end", "gpu_start");
    auto gpuElapsed = finalTask->consumedTime("gpu_end", "gpu_start");
    auto cpuElapsed = finalTask->consumedTime("cpu_end", "cpu_start");
    auto others = totalElapsed - gpuElapsed - cpuElapsed;

    if (verbose)
        std::cout << "Frame latency: " << frameLatency << "us, Elapsed: " << totalElapsed << "us, GPU: " << gpuElapsed << "us, CPU: " << cpuElapsed << "us, Others: " << others << "us" << std::endl;

    sender.send(finalTask->result);

    return {};
}

std::vector<std::shared_ptr<DecompressGpuTask>> DecompressCpuPipeline::work(std::shared_ptr<DecompressCpuTask> cpuTask, unsigned int threadId) {
    cpuTask->recordTimePoint("cpu_start");

    coder.decompress(cpuTask->input, cpuTask->memRefs[DecompressCpuTask::In]->host);

    cpuTask->recordTimePoint("cpu_end");

    return {std::make_shared<DecompressGpuTask>(
                std::move(cpuTask->timePoints),
                std::move(cpuTask->memRefs[DecompressCpuTask::In]),
                std::move(cpuTask->memRefs[DecompressCpuTask::Out])
            )};
}

std::vector<std::shared_ptr<DecompressFinalTask>> DecompressGpuPipeline::work(std::shared_ptr<DecompressGpuTask> gpuTask, unsigned int threadId) {
    gpuTask->recordTimePoint("gpu_start");
    workGpu(gpuTask);
    gpuTask->recordTimePoint("gpu_end");

    return {std::make_shared<DecompressFinalTask>(std::move(gpuTask->timePoints), std::move(gpuTask->memRefs[DecompressGpuTask::Out]))};
}

std::vector<std::shared_ptr<nullptr_t>> DecompressStatisticsPipeline::work(std::shared_ptr<DecompressFinalTask> finalTask, unsigned int threadId) {
    if (verbose) {
        auto totalElapsed = finalTask->consumedTime("gpu_end", "cpu_start");
        auto cpuElapsed = finalTask->consumedTime("cpu_end", "cpu_start");
        auto gpuElapsed = finalTask->consumedTime("gpu_end", "gpu_start");
        auto others = totalElapsed - cpuElapsed - gpuElapsed;
        std::cout << "Elapsed: " << totalElapsed << "us, CPU: " << cpuElapsed << "us, GPU: " << gpuElapsed
                  << "us, Others: " << others << "us" << std::endl;
    }

    completed++;
    if (completed == 500) {
        startTime = std::chrono::high_resolution_clock::now();
    } else if (completed % 500 == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - startTime).count();
        std::cout << "Completed: " << completed << ", FPS: " << (completed - 500) / duration << std::endl;
    }

    return {};
}

std::vector<std::shared_ptr<nullptr_t>> DecompressWindowPipeline::work(std::shared_ptr<DecompressFinalTask> finalTask, unsigned int threadId) {
    auto frameLatency = finalTask->consumedTime("gpu_end", "construct");
    auto totalElapsed = finalTask->consumedTime("gpu_end", "cpu_start");
    auto cpuElapsed = finalTask->consumedTime("cpu_end", "cpu_start");
    auto gpuElapsed = finalTask->consumedTime("gpu_end", "gpu_start");
    auto others = totalElapsed - gpuElapsed - cpuElapsed;

    if (verbose)
        std::cout << "Frame latency: " << frameLatency << "us, Elapsed: " << totalElapsed << "us, CPU: " << cpuElapsed << "us, GPU: " << gpuElapsed << "us, Others: " << others << "us" << std::endl;

    auto img = tensorToImg(finalTask->memRefs[DecompressFinalTask::Out]->host, height, width);

    if (!imagePath.empty())
        writeImgWithTimestamp(imagePath, img);

    //cv::imshow(windowName, img);
    //cv::waitKey(5);
    return {};
}