#include <iostream>

#include <opencv2/opencv.hpp>
#include <argparse/argparse.hpp>

#include "libs/network.h"
#include "libs/tensorrt_logger.h"
#include "libs/tensorrt_wrapper.h"
#include "libs/utils.h"
#include "libs/img_utils.h"
#include "pipelines/bmshj2018_factorized_pipelines.h"

const char *videoSrc = "nvarguscamerasrc ! "
                       "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, "
                       "format=(string)NV12, framerate=(fraction)%d/1 ! "
                       "nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! appsink";
const unsigned int batchSize = 1;

using namespace Bmshj2018Factorized::Pipelines;

struct Image : public ITask {
    cv::Mat img;

    explicit Image(cv::Mat img) : img(std::move(img)) {}
};

class ImageWriter : public PipelineImpl<Image> {
public:
    ImageWriter(std::string imagePath, unsigned int maxTasks) : imagePath(std::move(imagePath)), PipelineImpl<Image>(maxTasks, nullptr) {}
private:
    std::string imagePath;
    std::vector<std::shared_ptr<std::nullptr_t>> work(std::shared_ptr<Image> task, unsigned int threadId) override {
        if (!imagePath.empty()) writeImgWithTimestamp(imagePath, task->img);
        return {};
    }
};

int main(int argc, const char **argv) {
    argparse::ArgumentParser parser("cam_encoder");
    parser.add_description("CSI Camera encoder for bmshj2018_factorized");
    parser.add_argument("model").help("g_a.trt model file");
    parser.add_argument("coder_const").help("coder_const.bin constant file");
    parser.add_argument("-h", "--height").required().help("camera height").scan<'u', unsigned int>();
    parser.add_argument("-w", "--width").required().help("camera width").scan<'u', unsigned int>();
    parser.add_argument("-f", "--fps").required().help("camera fps").scan<'u', unsigned int>();
    parser.add_argument("-z", "--zero-copy").default_value(false).implicit_value(true).help("enable zero copy");
    parser.add_argument("-i", "--ip").required().help("recipient ip");
    parser.add_argument("-p", "--port").required().help("recipient port").scan<'u', unsigned int>();
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
    auto fps = parser.get<unsigned int>("--fps");
    bool verbose = parser.is_used("--verbose");
    auto memoryMode = parser.is_used("--zero-copy") ? ZERO_COPY : NORMAL;
    size_t outBlockSize = 192 * ceil(height / 16) * ceil(width / 16);

    TCPSender sender(parser.get("--ip"), parser.get<unsigned int>("--port"));
    cv::Mat img;

    TensorRTLogger logger(verbose);
    TensorRTWrapper trt(parser.get("model"), logger);
    {
        auto in = std::make_shared<MemoryWrapper<half>>(batchSize * 3 * height * width, memoryMode);
        auto outWarmup = std::make_shared<MemoryWrapper<half>>(batchSize * outBlockSize, memoryMode);
        trt.warmup(10, in, outWarmup);
    }

    auto gpuPipeline = std::make_shared<CompressGpuPipeline>(trt, 5);
    auto cpuPipeline = std::make_shared<CompressCpuPipeline>(width, height, parser.get("coder_const"), 5);
    auto senderPipeline = std::make_shared<CompressTCPSenderPipeline>(verbose, sender, 5);

    gpuPipeline->next(cpuPipeline)->next(senderPipeline);

    gpuPipeline->executeAsync(1);
    cpuPipeline->executeAsync(1);
    senderPipeline->executeAsync(1);

    cv::VideoCapture capture(stringFormat(videoSrc, width, height, fps), cv::CAP_GSTREAMER);
    if (!capture.isOpened()) {
        std::cerr << "Unable to open camera!" << std::endl;
        return 1;
    }

    std::string imagePath =  parser.is_used("--write-image") ? parser.get("--write-image") : "";
    auto imageWriter = std::make_shared<ImageWriter>(imagePath, 5);
    if (!imagePath.empty()) imageWriter->executeAsync(1);

    for (;;) {
        capture.read(img);
        if (img.size().width != width || img.size().height != height) {
            throw std::runtime_error("Invalid image from camera");
        }

        auto in = std::make_shared<MemoryWrapper<half>>(batchSize * 3 * height * width, memoryMode, HOST_TO_DEVICE);
        imgToTensor(img, in->host);
        auto out = std::make_shared<MemoryWrapper<half>>(batchSize * outBlockSize, memoryMode, DEVICE_TO_HOST);
        auto task = std::make_shared<CompressGpuTask>(in, out, outBlockSize, batchSize);
        bool result = gpuPipeline->tryFeedTask(task);
        if (!result) {
            std::cout << "GPU can't keep up! Dropping frame" << std::endl;
        }

        if (!imagePath.empty()) imageWriter->feedTask(std::make_shared<Image>(img));
    }

    return 0;
}