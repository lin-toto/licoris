#include "tensorrt_wrapper.h"

TensorRTWrapper::TensorRTWrapper(const std::string &modelPath, nvinfer1::ILogger& logger, bool synchronized)
: logger(logger), synchronized(synchronized) {
    std::ifstream modelFile(modelPath, std::ios::binary);
    if (!modelFile.good()) {
        throw std::runtime_error("Error reading model file");
    }

    modelFile.seekg(0, std::ifstream::end);
    size_t fileSize = modelFile.tellg();
    modelFile.seekg(0, std::ifstream::beg);

    char *buff = new char[fileSize];
    modelFile.read(buff, fileSize);

    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buff, fileSize));
    context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    delete[] buff;
}

TensorRTWrapper::~TensorRTWrapper() {
    cudaStreamDestroy(stream);
}

void TensorRTWrapper::doExecute(void **bindings) {
    context->enqueueV2(bindings, stream, nullptr);
    if (synchronized) cudaStreamSynchronize(stream);
}

void TensorRTWrapper::setInputDimension(int index, nvinfer1::Dims dims) {
    context->setBindingDimensions(index, dims);
}

nvinfer1::Dims TensorRTWrapper::getOutputDimension(int index) {
    return context->getBindingDimensions(index);
}
