#ifndef EMBEDLIC_CONSTANT_MAP_H
#define EMBEDLIC_CONSTANT_MAP_H

#include <cctype>
#include <cstddef>
#include <fstream>
#include <string>
#include <stdexcept>

#ifdef __linux__
#include <arpa/inet.h>
#elif _WIN32
#include <winsock.h>
#endif

template <typename T>
struct ConstantField {
    friend class ConstantMap;
public:
    size_t size;
    T *data;

    ConstantField(const ConstantField&) = delete;
    ConstantField& operator=(ConstantField) = delete;

    ~ConstantField() {
        delete[] data;
    }
protected:
    ConstantField(size_t size, T *data): size(size), data(data) {}
};

struct ConstantMap {
public:
    explicit ConstantMap(const std::string& fileName) : file(fileName, std::ios::binary) {
        if (!file.good())
            throw std::runtime_error("Unable to load constant file");
    }

    ConstantMap(const ConstantMap&) = delete;
    ConstantMap& operator=(ConstantMap) = delete;
protected:
    template <typename T, typename = std::enable_if_t<sizeof(T) == 4>> // Only support 4 bytes per item for now
    ConstantField<T> read(size_t checkSize = 0) {
        uint32_t magicBuf;
        file.read(reinterpret_cast<char *>(&magicBuf), 4);
        if (ntohl(magicBuf) != MAGIC) {
            throw std::runtime_error("Bad binary");
        }

        uint32_t sizeBuf;
        file.read(reinterpret_cast<char *>(&sizeBuf), 4);
        size_t size = ntohl(sizeBuf);
        if (checkSize != 0 && size != checkSize) {
            throw std::runtime_error("Bad size in binary sector");
        }

        T *dataBuf = new T[size];
        file.read(reinterpret_cast<char *>(dataBuf), size * sizeof(T));
        for (size_t i = 0; i < size; i++) {
            uint32_t flippedValue = ntohl(reinterpret_cast<uint32_t&>(dataBuf[i]));
            dataBuf[i] = reinterpret_cast<T&>(flippedValue);
        }

        return ConstantField<T>(size, dataBuf);
    }

    inline void assertEof() {
        file.get();
        if (!file.eof())
            throw std::runtime_error("Did not reach end of binary file");
    }
private:
    const uint32_t MAGIC = 0x00114514;

    std::ifstream file;
};

#endif //EMBEDLIC_CONSTANT_MAP_H
