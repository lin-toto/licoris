/* Copyright 2020 InterDigital Communications, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "entropy_coder/rans64.h"
#include <vector>
#include <string>

struct RansSymbol {
    uint16_t start;
    uint16_t range;
    bool bypass; // bypass flag to write raw bits to the stream
};

class CdfParamsAwareEncoderDecoder {
public:
    CdfParamsAwareEncoderDecoder(
            const int32_t *cdfs, const int32_t *cdfs_sizes, const int32_t *offsets,
            const size_t cdfs_2nddim_size
            ): cdfs(cdfs), cdfs_sizes(cdfs_sizes), offsets(offsets), cdfs_2nddim_size(cdfs_2nddim_size) {}

    CdfParamsAwareEncoderDecoder(const CdfParamsAwareEncoderDecoder &) = delete;
    CdfParamsAwareEncoderDecoder(CdfParamsAwareEncoderDecoder &&) = delete;
    CdfParamsAwareEncoderDecoder &operator=(const CdfParamsAwareEncoderDecoder &) = delete;
    CdfParamsAwareEncoderDecoder &operator=(CdfParamsAwareEncoderDecoder &&) = delete;

    std::string encode(const std::vector<int32_t> &symbols, const std::vector<int32_t> &indexes);
    std::vector<int32_t> decode(const std::string &encoded, const std::vector<int32_t> &indexes);

    void set_stream(std::string encoded);
    std::vector<int32_t> decode_stream(const std::vector<int32_t> &indexes);
    void release_stream();

private:
    const int32_t *cdfs;
    const int32_t *cdfs_sizes;
    const int32_t *offsets;
    size_t cdfs_2nddim_size;

    Rans64State _rans;
    std::string _stream;
    uint32_t *_ptr;
};
