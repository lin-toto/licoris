#ifndef EMBEDLIC_ENTROPY_CODER_H
#define EMBEDLIC_ENTROPY_CODER_H

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>

#ifndef NO_GPU_LIBRARIES
#include "cuda_fp16.h"
#endif

#include "entropy_coder/rans_interface.hpp"
#include "coder_constant_maps.h"

class EntropyCoder {
public:
    EntropyCoder(int c, int h, int w,
                 const int32_t *offsets, const int32_t *cdfLengths, const float *medians,
                 const int32_t *quantizedCdfs, int32_t cdf2ndDimSize)
                 : c(c), h(h), w(w), medians(medians),
                 coderImpl(quantizedCdfs, cdfLengths, offsets, cdf2ndDimSize) {}

#ifndef NO_GPU_LIBRARIES
    std::string compress(const half *input, const std::vector<int>& indexes);
    void decompress(const std::string &input, half *output, const std::vector<int>& indexes);
#endif

    std::string compress(const std::vector<int32_t> &input, const std::vector<int>& indexes);
    std::vector<int32_t> decompress(const std::string &input, const std::vector<int>& indexes);
    void quantize(std::vector<int>& input);
    void dequantize(std::vector<int>& input);

    CdfParamsAwareEncoderDecoder coderImpl;
protected:
    const float *medians;
    int c, h, w;

#ifndef NO_GPU_LIBRARIES
    std::vector<int> quantize(const half *input);
    void dequantize(const std::vector<int> &input, half *output);
#endif
};

class EntropyBottleneck : public EntropyCoder {
public:
    EntropyBottleneck(int c, int h, int w, std::shared_ptr<EntropyBottleneckConstantMap> map)
        : EntropyCoder(c, h, w,
                       map->offsets.data, map->cdfLengths.data,
                       map->medians.size > 0 ? map->medians.data : nullptr,
                       map->quantizedCdfs.data, map->cdf2ndDimSize.data[0]),
          map(std::move(map)) {
        buildIndexes();
    }

#ifndef NO_GPU_LIBRARIES
    inline std::string compress(const half *input) { return EntropyCoder::compress(input, indexes); }
    inline void decompress(const std::string &input, half *output) { return EntropyCoder::decompress(input, output, indexes); }
#endif

    inline std::string compress(const std::vector<int32_t> &input) { return EntropyCoder::compress(input, indexes); }
    inline std::vector<int32_t> decompress(const std::string &input) { return EntropyCoder::decompress(input, indexes); }
private:
    [[maybe_unused]] std::shared_ptr<EntropyBottleneckConstantMap> map;

    std::vector<int> indexes;
    void buildIndexes();
};

class GaussianConditional : public EntropyCoder {
public:
    GaussianConditional(int c, int h, int w, float scaleBound, std::shared_ptr<GaussianConditionalConstantMap> map)
            : EntropyCoder(c, h, w,
                           map->offsets.data, map->cdfLengths.data,
                           map->medians.size > 0 ? map->medians.data : nullptr,
                           map->quantizedCdfs.data, map->cdf2ndDimSize.data[0]),
              scaleBound(scaleBound), map(std::move(map)) {}

#ifndef NO_GPU_LIBRARIES
    std::vector<int> buildIndexes(const half *scales);
#endif
private:
    std::shared_ptr<GaussianConditionalConstantMap> map;
    float scaleBound;
};

class CModelGaussianConditional : public EntropyCoder {
public:
    CModelGaussianConditional(int c, int h, int w,
                              float scaleMin, float scaleMax, int meanQL, int scaleQL, int ql, int cdfYRange,
                              std::shared_ptr<CModelGaussianConditionalConstantMap> map)
            : EntropyCoder(c, h, w,
                           map->offsets.data, map->cdfLengths.data,
                           map->medians.size > 0 ? map->medians.data : nullptr,
                           map->quantizedCdfs.data, map->cdf2ndDimSize.data[0]),
             scaleMin(scaleMin), scaleMax(scaleMax), meanQL(meanQL), scaleQL(scaleQL), ql(ql), cdfYRange(cdfYRange),
             map(std::move(map)) {
        scaleHalfPointMin = calculateScaleHalfPoint(0.5f);
        scaleHalfPointMax = calculateScaleHalfPoint(ql - 0.5f);
    }

    std::vector<int> quantize(const int8_t *input, const int16_t *means);
    void dequantize(const std::vector<int>& input, int8_t *output, const int16_t *means);
    std::vector<int> buildIndexes(const int16_t *means, const int16_t *scales);
private:
    [[maybe_unused]] std::shared_ptr<CModelGaussianConditionalConstantMap> map;

    float scaleMin, scaleMax;
    int scaleHalfPointMin, scaleHalfPointMax;
    int ql, meanQL, scaleQL;
    int cdfYRange;

    [[nodiscard]] int calculateScaleHalfPoint(float idx) const;
};

#endif //EMBEDLIC_ENTROPY_CODER_H
