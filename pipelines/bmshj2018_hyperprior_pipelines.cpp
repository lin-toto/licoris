#include "bmshj2018_hyperprior_pipelines.h"

using namespace Bmshj2018Hyperprior::Pipelines;

std::vector<std::shared_ptr<CompressCpuTask>> CompressGpuPipeline::work(std::shared_ptr<CompressGpuTask> gpuTask, unsigned int threadId) {
    gpuTask->recordTimePoint("gpu_start");
    workGpu(gpuTask);
    gpuTask->recordTimePoint("gpu_end");

    std::vector<std::shared_ptr<CompressCpuTask>> result;
    result.reserve(gpuTask->batchSize);
    for (unsigned int i = 0; i < gpuTask->batchSize; i++) {
        result.push_back(std::make_shared<CompressCpuTask>(
            gpuTask->timePoints,
            gpuTask->memRefs[CompressGpuTask::YOut],
            gpuTask->memRefs[CompressGpuTask::ZOut],
            gpuTask->memRefs[CompressGpuTask::ScalesOut],
            i * gpuTask->yBlockSize, i * gpuTask->zBlockSize, i * gpuTask->scalesBlockSize)
        );
    }

    return result;
}

std::vector<std::shared_ptr<CompressFinalTask>> CompressCpuPipeline::work(std::shared_ptr<CompressCpuTask> cpuTask, unsigned int threadId) {
    cpuTask->recordTimePoint("cpu_start");

    std::string zStrings = entropyBottleneck.compress(cpuTask->memRefs[CompressCpuTask::ZOut]->host + cpuTask->zOffset);
    auto indexes = gaussianConditional.buildIndexes(cpuTask->memRefs[CompressCpuTask::ScalesOut]->host + cpuTask->scalesOffset);
    std::string yStrings = gaussianConditional.compress(cpuTask->memRefs[CompressCpuTask::YOut]->host + cpuTask->yOffset, indexes);

    cpuTask->recordTimePoint("cpu_end");

    return {std::make_shared<CompressFinalTask>(
                cpuTask->timePoints, std::move(yStrings), std::move(zStrings)
            )};
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

std::vector<std::shared_ptr<DecompressGeneralTask>> DecompressZHatPipeline::work(std::shared_ptr<DecompressGeneralTask> generalTask, unsigned int threadId) {
    generalTask->recordTimePoint("zhat_start");

    entropyBottleneck.decompress(generalTask->zStrings, generalTask->memRefs[DecompressGeneralTask::ZIn]->host);
    generalTask->zStrings = "";

    generalTask->recordTimePoint("zhat_end");
    return {std::move(generalTask)};
}

std::vector<std::shared_ptr<DecompressGeneralTask>> DecompressScalesPipeline::work(std::shared_ptr<DecompressGeneralTask> generalTask, unsigned int threadId) {
    generalTask->recordTimePoint("scales_start");

    trt.copyToDevice(generalTask->memRefs[DecompressGeneralTask::ZIn]);
    trt.execute(generalTask->memRefs[DecompressGeneralTask::ZIn], generalTask->memRefs[DecompressGeneralTask::ScalesOut]);
    trt.copyToHost(generalTask->memRefs[DecompressGeneralTask::ScalesOut]);
    trt.sync();

    generalTask->recordTimePoint("scales_end");

    return {std::move(generalTask)};
}

std::vector<std::shared_ptr<DecompressGeneralTask>> DecompressGaussianDecodePipeline::work(std::shared_ptr<DecompressGeneralTask> generalTask, unsigned int threadId) {
    generalTask->recordTimePoint("gaussian_start");

    auto indexes = gaussianConditional.buildIndexes(generalTask->memRefs[DecompressGeneralTask::ScalesOut]->host);
    gaussianConditional.decompress(generalTask->yStrings, generalTask->memRefs[DecompressGeneralTask::YIn]->host, indexes);

    generalTask->recordTimePoint("gaussian_end");
    return {std::move(generalTask)};
}

std::vector<std::shared_ptr<DecompressFinalTask>> DecompressYHatPipeline::work(std::shared_ptr<DecompressGeneralTask> generalTask, unsigned int threadId) {
    generalTask->recordTimePoint("yhat_start");

    trt.copyToDevice(generalTask->memRefs[DecompressGeneralTask::YIn]);
    trt.execute(generalTask->memRefs[DecompressGeneralTask::YIn], generalTask->memRefs[DecompressGeneralTask::XOut]);
    trt.copyToHost(generalTask->memRefs[DecompressGeneralTask::XOut]);
    trt.sync();

    generalTask->recordTimePoint("yhat_end");

    return {std::make_shared<DecompressFinalTask>(
            generalTask->timePoints, std::move(generalTask->memRefs[DecompressGeneralTask::XOut]))};
}

std::vector<std::shared_ptr<nullptr_t>> DecompressStatisticsPipeline::work(std::shared_ptr<DecompressFinalTask> finalTask, unsigned int threadId) {
    if (verbose) {
        auto totalElapsed = finalTask->consumedTime("yhat_end", "zhat_start");
        auto zHatElapsed = finalTask->consumedTime("zhat_end", "zhat_start");
        auto scalesElapsed = finalTask->consumedTime("scales_end", "scales_start");
        auto gaussianElapsed = finalTask->consumedTime("gaussian_end", "gaussian_start");
        auto yHatElapsed = finalTask->consumedTime("yhat_end", "yhat_start");
        auto others = totalElapsed - zHatElapsed - scalesElapsed - gaussianElapsed - yHatElapsed;
        std::cout << "Elapsed: " << totalElapsed << "us, zHat: " << zHatElapsed << "us, scales: " << scalesElapsed
                 << "us, gaussian: " << gaussianElapsed << "us, yHat: " << yHatElapsed
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