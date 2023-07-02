#include "entropy_coder.h"

#include <cmath>

std::string EntropyCoder::compress(const std::vector<int32_t> &input, const std::vector<int> &indexes) {
    return coderImpl.encode(input, indexes);
}


std::vector<int32_t> EntropyCoder::decompress(const std::string &input, const std::vector<int> &indexes) {
    return coderImpl.decode(input, indexes);
}

void EntropyBottleneck::buildIndexes() {
    indexes.reserve(c * h * w);

    for (int _c = 0; _c < c; _c++) {
        for (int _h = 0; _h < h; _h++) {
            for (int _w = 0; _w < w; _w++) {
                indexes.push_back(_c);
            }
        }
    }
}

void EntropyCoder::quantize(std::vector<int>& input) {
    if (medians == nullptr) return;

    for (int _c = 0; _c < c; _c++) {
        for (int _h = 0; _h < h; _h++) {
            for (int _w = 0; _w < w; _w++) {
                int index = _c * (h * w) + _h * w + _w;
                input[index] -= medians[_c];
            }
        }
    }
}

void EntropyCoder::dequantize(std::vector<int>& input) {
    if (medians == nullptr) return;
    for (int _c = 0; _c < c; _c++) {
        for (int _h = 0; _h < h; _h++) {
            for (int _w = 0; _w < w; _w++) {
                int index = _c * (h * w) + _h * w + _w;
                input[index] += medians[_c];
            }
        }
    }
}

#ifndef NO_GPU_LIBRARIES
std::string EntropyCoder::compress(const half *input, const std::vector<int> &indexes) {
    std::vector<int> symbols = quantize(input);
    return coderImpl.encode(symbols, indexes);
}

void EntropyCoder::decompress(const std::string &input, half *output, const std::vector<int> &indexes) {
    std::vector<int> symbols = coderImpl.decode(input, indexes);
    dequantize(symbols, output);
}

std::vector<int> EntropyCoder::quantize(const half *input) {
    std::vector<int> result;
    result.reserve(c * h * w);

    for (int _c = 0; _c < c; _c++) {
        for (int _h = 0; _h < h; _h++) {
            for (int _w = 0; _w < w; _w++) {
                int index = _c * (h * w) + _h * w + _w;
                int symbol;
                if (medians == nullptr) {
                    symbol = static_cast<int>(std::round(__half2float(input[index])));
                } else {
                    symbol = static_cast<int>(std::round(__half2float(input[index]) - medians[_c]));
                }

                result.push_back(symbol);
            }
        }
    }

    return result;
}

void EntropyCoder::dequantize(const std::vector<int> &input, half *output) {
    for (int _c = 0; _c < c; _c++) {
        for (int _h = 0; _h < h; _h++) {
            for (int _w = 0; _w < w; _w++) {
                int index = _c * (h * w) + _h * w + _w;
                if (medians == nullptr) {
                    output[index] = __float2half(input[index]);
                } else {
                    output[index] = __float2half(input[index] + medians[_c]);
                }
            }
        }
    }
}

std::vector<int> GaussianConditional::buildIndexes(const half *scales) {
    std::vector<int> indexes;
    indexes.reserve(c * h * w);

    for (int i = 0; i < c * h * w; i++) {
        float scale = std::max(__half2float(scales[i]), scaleBound);
        int index = std::lower_bound(map->scaleTable.data, map->scaleTable.data + map->scaleTable.size - 1, scale) - map->scaleTable.data;
        indexes.push_back(index);
    }

    return indexes;
}
#endif

std::vector<int> CModelGaussianConditional::quantize(const int8_t *input, const int16_t *means) {
    std::vector<int> result;
    result.reserve(c * h * w);

    for (int i = 0; i < c * h * w; i++) {
        // The -cdfYRange offset should be handled in the constant map
        result.push_back(std::clamp(static_cast<int>(input[i]) - static_cast<int>(std::ceil(means[i] / static_cast<float>(meanQL))), -cdfYRange, cdfYRange));
    }

    return result;
}

void CModelGaussianConditional::dequantize(const std::vector<int> &input, int8_t *output, const int16_t *means) {
    for (int i = 0; i < c * h * w; i++) {
        output[i] = static_cast<int>(input[i]) + static_cast<int>(std::ceil(means[i] / static_cast<float>(meanQL)));
    }
}

std::vector<int> CModelGaussianConditional::buildIndexes(const int16_t *means, const int16_t *scales) {
    std::vector<int> indexes;
    indexes.reserve(c * h * w);

    for (int i = 0; i < c * h * w; i++) {
        auto scale = std::abs(scales[i]);
        int index;
        if (scale < scaleHalfPointMin) {
            index = 0;
        } else if (scale >= scaleHalfPointMax) {
            index = ql - 1;
        } else {
            index = map->scaleLut.data[scale - scaleHalfPointMin];
        }

        index += ((means[i] % meanQL + meanQL) % meanQL) * ql;
        indexes.push_back(index);
    }

    return indexes;
}

int CModelGaussianConditional::calculateScaleHalfPoint(float idx) const {
    float logRange = std::log(scaleMax) - std::log(scaleMin);
    float logStep = logRange / static_cast<float>(ql - 1);

    float scale = std::exp(std::log(scaleMin) + (idx * logStep));
    return static_cast<int>(std::round(scale * scaleQL));
}