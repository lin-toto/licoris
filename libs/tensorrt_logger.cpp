//
// Created by Toto Lin on 2021/10/24.
//

#include "tensorrt_logger.h"

void TensorRTLogger::log(Severity severity, const nvinfer1::AsciiChar *msg) noexcept {
    if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
        std::cerr << "[TensorRT] " << msg << std::endl;
    } else {
        if (verbose) std::cout << "[TensorRT] " << msg << std::endl;
    }
}

