#ifndef EMBEDLIC_TENSORRT_LOGGER_H
#define EMBEDLIC_TENSORRT_LOGGER_H

#include <NvInfer.h>
#include <iostream>

class TensorRTLogger : public nvinfer1::ILogger {
public:
    explicit TensorRTLogger(bool verbose = false) : verbose(verbose) {}
    void log(Severity severity, const nvinfer1::AsciiChar *msg) noexcept override;
private:
    bool verbose;
};


#endif //EMBEDLIC_TENSORRT_LOGGER_H
