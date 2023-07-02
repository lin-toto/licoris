#include "img_utils.h"

cv::Mat readImg(const std::string& path) {
    cv::Mat img = cv::imread(path);
    if (!img.data) {
        throw std::runtime_error("Unable to read image file");
    }

    return img;
}

void writeImgWithTimestamp(const std::string& basePath, cv::Mat img) {
    uint64_t timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::string fileName = basePath + "/" + std::to_string(timestamp) + ".png";
    cv::imwrite(fileName, img);
}

void imgToFpgaFormat(cv::Mat img, uint8_t *out, int n, int c, int h, int w, int channelsPerGroup, int heightPerGroup) {
    if (c != 3)
        throw std::runtime_error("Image must have channel number of 3");

    if (c % channelsPerGroup != 0 || h % heightPerGroup != 0)
        throw std::runtime_error("Invalid number of channels or height per group");

    int i = 0;
    for (int _n = 0; _n < n; _n++) {
        for (int heightGroup = 0; heightGroup < h / heightPerGroup; heightGroup++) {
            for (int channelGroup = 0; channelGroup < c / channelsPerGroup; channelGroup++) {
                for (int heightOffset = 0; heightOffset < heightPerGroup; heightOffset++) {
                    for (int _w = 0; _w < w; _w++) {
                        for (int channelOffset = 0; channelOffset < channelsPerGroup; channelOffset++) {
                            int _h = heightGroup * heightPerGroup + heightOffset;
                            int _c = channelGroup * channelsPerGroup + channelOffset;

                            //std::cout << _n << " " << _c << " " << _h << " " << _w << std::endl;

                            out[i] = img.at<cv::Vec3b>(_h, _w)[2 - _c];
                            i++;
                        }
                    }
                }
            }
        }
    }
}

#ifndef NO_GPU_LIBRARIES
void absData(const half *in, half *out, int n, int c, int h, int w) {
    for (int i = 0; i < n * c * h * w; i++) {
        float tmp = __half2float(in[i]);
        out[i] = __float2half(tmp > 0 ? tmp : -tmp);
    }
}

void debugPrintTensor(const half *tensor, int n, int c, int h, int w) {
    for (int _n = 0; _n < n; _n++) {
        for (int _c = 0; _c < c; _c++) {
            for (int _h = 0; _h < h; _h++) {
                for (int _w = 0; _w < w; _w++) {
                    printf("%.4f ", __half2float(tensor[_n * c * h * w + _c * h * w + _h * w + _w]));
                }
                puts("");
            }
            puts("");
        }
        puts("");
    }

    fflush(stdout);
};

void readImgCHW(const std::string& path, half *out, unsigned int replicas) {
    cv::Mat img = readImg(path);
    imgToTensor(img, out, replicas);
}

void imgToTensor(cv::Mat img, half* out, unsigned int replicas) {
    int height = img.rows, width = img.cols;

#pragma omp parallel for
#pragma omp unroll partial(4)
    for (unsigned i = 0; i < height; i++) {
        for (unsigned j = 0; j < width; j++) {
            cv::Vec3b bgr = img.at<cv::Vec3b>(i, j);
            out[0 * height * width + i * width + j] = __float2half(1.0 / 255.0 * bgr[2]);
            out[1 * height * width + i * width + j] = __float2half(1.0 / 255.0 * bgr[1]);
            out[2 * height * width + i * width + j] = __float2half(1.0 / 255.0 * bgr[0]);
        }
    }

    for (int i = 1; i < replicas; i++) {
        size_t imageSize = 3 * height * width;
        std::copy(out, out + imageSize, out + i * imageSize);
    }
}

cv::Mat tensorToImg(const half *in, unsigned int height, unsigned int width) {
    cv::Mat img(height, width, CV_8UC3);

#pragma omp parallel for
#pragma omp unroll partial(4)
    for (unsigned i = 0; i < height; i++) {
        for (unsigned j = 0; j < width; j++) {
            float r = std::clamp(__half2float(in[0 * height * width + i * width + j]), 0.0f, 1.0f);
            float g = std::clamp(__half2float(in[1 * height * width + i * width + j]), 0.0f, 1.0f);
            float b = std::clamp(__half2float(in[2 * height * width + i * width + j]), 0.0f, 1.0f);

            img.at<cv::Vec3b>(i, j)[2] = static_cast<int>(r * 255.0);
            img.at<cv::Vec3b>(i, j)[1] = static_cast<int>(g * 255.0);
            img.at<cv::Vec3b>(i, j)[0] = static_cast<int>(b * 255.0);
        }
    }

    return img;
}
#endif