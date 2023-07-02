#include <iostream>

#include <opencv2/opencv.hpp>
#include <argparse/argparse.hpp>

#include "libs/network.h"
#include "libs/tensorrt_logger.h"
#include "libs/tensorrt_wrapper.h"
#include "libs/utils.h"
#include "libs/img_utils.h"
#include "pipelines/bmshj2018_factorized_pipelines.h"

const unsigned int batchSize = 1;

using namespace Bmshj2018Factorized::Pipelines;

int main(int argc, const char **argv) {
    argparse::ArgumentParser parser("cam_decoder");
    parser.add_description("CSI Camera decoder for bmshj2018_factorized");
    parser.add_argument("model").help("g_s.trt model file");
    parser.add_argument("coder_const").help("coder_const.bin constant file");
    parser.add_argument("-h", "--height").required().help("camera height").scan<'u', unsigned int>();
    parser.add_argument("-w", "--width").required().help("camera width").scan<'u', unsigned int>();
    parser.add_argument("-z", "--zero-copy").default_value(false).implicit_value(true).help("enable zero copy");
    parser.add_argument("-i", "--ip").required().help("bind ip");
    parser.add_argument("-p", "--port").required().help("bind port").scan<'u', unsigned int>();
    parser.add_argument("-v", "--verbose").help("verbose output").default_value(false).implicit_value(true);
    parser.add_argument("--write-image").help("write image captured with timestamp to this path").default_value("");

    try {
        parser.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << "Error: " << err.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    auto height = parser.get<unsigned int>("--height");
    auto width = parser.get<unsigned int>("--width");
    bool verbose = parser.is_used("--verbose");
    auto memoryMode = parser.is_used("--zero-copy") ? ZERO_COPY : NORMAL;

    TCPReceiver receiver(parser.get("--ip"), parser.get<unsigned int>("--port"));
    cv::Mat img;

    TensorRTLogger logger(verbose);
    TensorRTWrapper trt(parser.get("model"), logger);
    {
        auto in = std::make_shared<MemoryWrapper<half>>(batchSize * 192 * ceil(height / 16) * ceil(width / 16), memoryMode);
        auto outWarmup = std::make_shared<MemoryWrapper<half>>(batchSize * 3 * height * width, memoryMode);
        trt.warmup(10, in, outWarmup);
    }

    std::string imagePath = parser.is_used("--write-image") ? parser.get("--write-image") : "";

    auto cpuPipeline = std::make_shared<DecompressCpuPipeline>(width, height, parser.get("coder_const"), 10);
    auto gpuPipeline = std::make_shared<DecompressGpuPipeline>(trt, 10);
    auto finalPipeline = std::make_shared<DecompressWindowPipeline>(width, height, verbose, std::move(imagePath), 10);

    cpuPipeline->next(gpuPipeline)->next(finalPipeline);

    cpuPipeline->executeAsync(2);
    gpuPipeline->executeAsync(1);
    finalPipeline->executeAsync(1);

    receiver.waitClient();
    std::cout << "Client is connected!" << std::endl;
    for (;;) {
        std::string strings = receiver.receive();

        auto in = std::make_shared<MemoryWrapper<half>>(batchSize * 192 * ceil(height / 16) * ceil(width / 16), memoryMode, HOST_TO_DEVICE);
        auto out = std::make_shared<MemoryWrapper<half>>(batchSize * 3 * height * width, memoryMode, DEVICE_TO_HOST);
        auto task = std::make_shared<DecompressCpuTask>(in, out, std::move(strings));
        bool result = cpuPipeline->tryFeedTask(task);
        if (!result) {
            std::cout << "Can't keep up! Dropping frame" << std::endl;
        }
    }

    return 0;
}