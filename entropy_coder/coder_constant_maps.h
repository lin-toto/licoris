#ifndef EMBEDLIC_CODER_CONSTANT_MAPS_H
#define EMBEDLIC_CODER_CONSTANT_MAPS_H

#include <cctype>
#include "libs/constant_map.h"

struct EntropyBottleneckConstantMap : public ConstantMap {
public:
    ConstantField<int32_t> offsets;
    ConstantField<int32_t> cdfLengths;
    ConstantField<float> medians;
    ConstantField<int32_t> cdf2ndDimSize;
    ConstantField<int32_t> quantizedCdfs;

    explicit EntropyBottleneckConstantMap(const std::string& fileName, bool end = true) :
        ConstantMap(fileName),
        offsets(read<int32_t>()),
        cdfLengths(read<int32_t>()),
        medians(read<float>()),
        cdf2ndDimSize(read<int32_t>(1)),
        quantizedCdfs(read<int32_t>()) {
        if (end) assertEof();
    }
};

struct GaussianConditionalConstantMap : public EntropyBottleneckConstantMap {
public:
    ConstantField<float> scaleTable;

    explicit GaussianConditionalConstantMap(const std::string& fileName) :
        EntropyBottleneckConstantMap(fileName, false),
        scaleTable(read<float>()) {
        assertEof();
    }
};

struct CModelGaussianConditionalConstantMap : public EntropyBottleneckConstantMap {
public:
    ConstantField<int32_t> scaleLut;

    explicit CModelGaussianConditionalConstantMap(const std::string& fileName) :
            EntropyBottleneckConstantMap(fileName, false),
            scaleLut(read<int32_t>()) {
        assertEof();
    }
};


#endif //EMBEDLIC_CODER_CONSTANT_MAPS_H
