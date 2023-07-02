#include "libs/tensorrt_logger.h"
#include "libs/tensorrt_wrapper.h"
#include "pipelines/bmshj2018_factorized_pipelines.h"
#include "libs/utils.h"
#include "libs/img_utils.h"

#include <argparse/argparse.hpp>

using namespace Bmshj2018Factorized::Pipelines;

int main(int argc, const char **argv) {
    argparse::ArgumentParser parser("bmshj2018_factorized_compressor");
    parser.add_description("Compress evaluation for bmshj2018_factorized");
    parser.add_argument("model").help("g_a.trt model file");
    parser.add_argument("coder_const").help("coder_const.bin constant file");
    parser.add_argument("image").help("image file to evaluate");
    parser.add_argument("-b", "--batch").default_value(1).help("batch size").scan<'u', unsigned int>();
    parser.add_argument("-t", "--thread").default_value(2).help("threads for entropy coder").scan<'u', unsigned int>();
    parser.add_argument("-h", "--height").required().help("image height").scan<'u', unsigned int>();
    parser.add_argument("-w", "--width").required().help("image width").scan<'u', unsigned int>();
    parser.add_argument("-z", "--zero-copy").default_value(false).implicit_value(true).help("enable zero copy");
    parser.add_argument("-v", "--verbose").help("verbose output").default_value(false).implicit_value(true);

    try {
        parser.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << "Error: " << err.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    auto batchSize = parser.get<unsigned int>("--batch");
    auto height = parser.get<unsigned int>("--height");
    auto width = parser.get<unsigned int>("--width");
    bool verbose = parser.is_used("--verbose");
    auto memoryMode = parser.is_used("--zero-copy") ? ZERO_COPY : NORMAL;

    auto in = std::make_shared<MemoryWrapper<half>>(batchSize * 3 * height * width, memoryMode, HOST_TO_DEVICE);
    readImgCHW(parser.get("image"), in->host, batchSize);

    TensorRTLogger logger(verbose);
    TensorRTWrapper trt(parser.get("model"), logger, false);

    auto gpuPipeline = std::make_shared<CompressGpuPipeline>(trt, 5);
    auto cpuPipeline = std::make_shared<CompressCpuPipeline>(width, height, parser.get("coder_const"), 5);
    auto finalPipeline = std::make_shared<CompressStatisticsPipeline>(verbose, 5);

    gpuPipeline->next(cpuPipeline)->next(finalPipeline);

    gpuPipeline->executeAsync(1);
    cpuPipeline->executeAsync(parser.get<unsigned int>("--thread"));
    finalPipeline->executeAsync(1);

    for (;;) {
        size_t outBlockSize = 192 * ceil(height / 16) * ceil(width / 16);
        auto out = std::make_shared<MemoryWrapper<half>>(batchSize * outBlockSize, memoryMode, DEVICE_TO_HOST);
        auto task = std::make_shared<CompressGpuTask>(in, out, outBlockSize, batchSize);
        gpuPipeline->feedTask(task);
    }

    return 0;
}