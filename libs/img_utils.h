#ifndef EMBEDLIC_IMG_UTILS_H
#define EMBEDLIC_IMG_UTILS_H

#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#ifndef NO_GPU_LIBRARIES
#include <cuda_fp16.h>
#endif

#include "utils.h"
#include "memory.h"
#include <iostream>

cv::Mat readImg(const std::string& path);
void writeImgWithTimestamp(const std::string& basePath, cv::Mat img);

void imgToFpgaFormat(cv::Mat img, uint8_t *out, int n, int c, int h, int w, int channelsPerGroup, int heightsPerGroup);

template<typename T, typename U>
void fpgaScramble(const T *in, U *out, int n, int c, int h, int w, int channelsPerGroup, int heightsPerGroup) {
    if (h % heightsPerGroup != 0)
        throw std::runtime_error("Invalid number of channels or height per group");

    int channelGroupCount = std::ceil(c * 1.0f / channelsPerGroup); // May have extra channels at the end
    int i = 0;
    for (int _n = 0; _n < n; _n++) {
        for (int heightGroup = 0; heightGroup < h / heightsPerGroup; heightGroup++) {
            for (int channelGroup = 0; channelGroup < channelGroupCount; channelGroup++) {
                for (int heightOffset = 0; heightOffset < heightsPerGroup; heightOffset++) {
                    for (int _w = 0; _w < w; _w++) {
                        for (int channelOffset = 0; channelOffset < channelsPerGroup; channelOffset++) {
                            int _h = heightGroup * heightsPerGroup + heightOffset;
                            int _c = channelGroup * channelsPerGroup + channelOffset;

                            out[i] = _c < c ? in[_n * c * h * w + _c * h * w + _h * w + _w] : 0;
                            i++;
                        }
                    }
                }
            }
        }
    }
}

template<typename T, typename U>
void fpgaUnscramble(const T *in, U *out, int n, int c, int h, int w, int channelsPerGroup, int heightsPerGroup) {
    if (h % heightsPerGroup != 0)
        throw std::runtime_error("Invalid number of height per group");

    int channelGroupCount = std::ceil(c * 1.0f / channelsPerGroup); // May have extra channels at the end
    int i = 0;
    for (int _n = 0; _n < n; _n++) {
        for (int heightGroup = 0; heightGroup < h / heightsPerGroup; heightGroup++) {
            for (int channelGroup = 0; channelGroup < channelGroupCount; channelGroup++) {
                for (int heightOffset = 0; heightOffset < heightsPerGroup; heightOffset++) {
                    for (int _w = 0; _w < w; _w++) {
                        for (int channelOffset = 0; channelOffset < channelsPerGroup; channelOffset++) {
                            int _h = heightGroup * heightsPerGroup + heightOffset;
                            int _c = channelGroup * channelsPerGroup + channelOffset;

                            if (_c < c) out[_n * c * h * w + _c * h * w + _h * w + _w] = in[i];
                            //else if (in[i] != 0) {
                            //    abort();
                            //    throw std::runtime_error("Expected 0 at empty data");
                            //
                            //                          }

                            i++;
                        }
                    }
                }
            }
        }
    }
}

#ifndef NO_GPU_LIBRARIES
void absData(const half *in, half *out, int n, int c, int h, int w);
void debugPrintTensor(const half *tensor, int n, int c, int h, int w);
void readImgCHW(const std::string& path, half *out, unsigned int replicas = 1);
void imgToTensor(cv::Mat img, half *out, unsigned int replicas = 1);
cv::Mat tensorToImg(const half *in, unsigned int height, unsigned int width);
#endif

template<typename T>
void padData(const T *in, T *out, int c, int h, int w, int padTop, int padBottom, int padLeft, int padRight, T padContent) {
    const T *inPtr = in;
    T *outPtr = out;
    for (int _c = 0; _c < c; _c++) {
        std::fill(outPtr, outPtr + padTop * (padLeft + w + padRight), padContent);
        outPtr += padTop * (padLeft + w + padRight);
        for (int _h = 0; _h < h; _h++) {
            std::fill(outPtr, outPtr + padLeft, padContent);
            outPtr += padLeft;

            std::copy(inPtr, inPtr + w, outPtr);
            inPtr += w;
            outPtr += w;

            std::fill(outPtr, outPtr + padLeft, padContent);
            outPtr += padRight;
        }
        std::fill(outPtr, outPtr + padBottom * (padLeft + w + padRight), padContent);
        outPtr += padBottom * (padLeft + w + padRight);
    }
}

template<typename T>
void cropData(const T *in, T *out, int c, int h, int w, int cropH, int cropW, int offsetH, int offsetW) {
    const T *inPtr;
    T *outPtr = out;
    for (int _c = 0; _c < c; _c++) {
        inPtr = in + _c * h * w + offsetH * w;
        for (int _h = 0; _h < cropH; _h++) {
            std::copy(inPtr + offsetW, inPtr + offsetW + cropW, outPtr);
            inPtr += w;
            outPtr += cropW;
        }
    }
}

template<typename T>
void pixelShuffle(const T *in, T *out, int c, int h, int w, int scale) {
    for (int _c = 0; _c < c; _c++) {
        int realC = _c / (scale * scale);
        for (int _h = 0; _h < h; _h++) {
            for (int _w = 0; _w < w; _w++) {
                int realH = _h * scale + _c % (scale * scale) / scale, realW = _w * scale + _c % scale;
                out[realC * h * scale * w * scale + realH * w * scale + realW] = in[_c * h * w + _h * w + _w];
            }
        }
    }
}

#endif //EMBEDLIC_IMG_UTILS_H
