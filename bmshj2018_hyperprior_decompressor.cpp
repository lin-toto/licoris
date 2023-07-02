#include "libs/tensorrt_logger.h"
#include "libs/tensorrt_wrapper.h"
#include "pipelines/bmshj2018_hyperprior_pipelines.h"
#include "libs/utils.h"
#include "libs/img_utils.h"

#include <argparse/argparse.hpp>

using namespace Bmshj2018Hyperprior::Pipelines;

int main(int argc, const char **argv) {
    argparse::ArgumentParser parser("bmshj2018_hyperprior_decompressor");
    parser.add_description("Decompress evaluation for bmshj2018_hyperprior");
    parser.add_argument("g_s").help("g_s.trt model file");
    parser.add_argument("h_s").help("h_s.trt model file");
    parser.add_argument("bottleneck_const").help("bottleneck_const.bin constant file");
    parser.add_argument("gaussian_const").help("gaussian_const.bin constant file");
    parser.add_argument("y_strings").help("y_strings.txt file to evaluate");
    parser.add_argument("z_strings").help("z_strings.txt file to evaluate");
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
    if (batchSize != 1) {
        std::cerr << "Error: batch_size != 1 is not supported";
        return 1;
    }

    std::string yStrings = readFile(parser.get("y_strings"));
    std::string zStrings = readFile(parser.get("z_strings"));

    TensorRTLogger logger(verbose);
    TensorRTWrapper trtGs(parser.get("g_s"), logger, false);
    TensorRTWrapper trtHs(parser.get("h_s"), logger, false);

    size_t xBlockSize = 3 * height * width;
    size_t yBlockSize = 192 * ceil(height / 16) * ceil(width / 16);
    size_t zBlockSize = 128 * ceil(height / 64) * ceil(width / 64);
    size_t scalesBlockSize = 192 * ceil(height / 16) * ceil(width / 16);

    auto zHatPipeline = std::make_shared<DecompressZHatPipeline>(width, height, parser.get("bottleneck_const"), 3);
    auto scalesPipeline = std::make_shared<DecompressScalesPipeline>(trtHs, 3);
    auto gaussianPipeline = std::make_shared<DecompressGaussianDecodePipeline>(width, height, parser.get("gaussian_const"), 3);
    auto yHatPipeline = std::make_shared<DecompressYHatPipeline>(trtGs, 3);
    auto finalPipeline = std::make_shared<DecompressStatisticsPipeline>(verbose, 3);

    zHatPipeline->next(scalesPipeline)->next(gaussianPipeline)->next(yHatPipeline)->next(finalPipeline);

    zHatPipeline->executeAsync(1);
    scalesPipeline->executeAsync(1);
    gaussianPipeline->executeAsync(parser.get<unsigned int>("--thread"));
    yHatPipeline->executeAsync(1);
    finalPipeline->executeAsync(1);

    for (;;) {
        auto x = std::make_shared<MemoryWrapper<half>>(batchSize * xBlockSize, memoryMode, DEVICE_TO_HOST);
        auto y = std::make_shared<MemoryWrapper<half>>(batchSize * yBlockSize, memoryMode, HOST_TO_DEVICE);
        auto z = std::make_shared<MemoryWrapper<half>>(batchSize * zBlockSize, memoryMode, HOST_TO_DEVICE);
        auto scales = std::make_shared<MemoryWrapper<half>>(batchSize * scalesBlockSize, memoryMode, DEVICE_TO_HOST);

        auto task = std::make_shared<DecompressGeneralTask>(yStrings, zStrings, z, scales, y, x);
        zHatPipeline->feedTask(task);
    }

    return 0;
}