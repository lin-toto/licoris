#include "utils.h"

void mySleep(int sec) {
#ifdef __linux__
    sleep(sec);
#elif _WIN32
    Sleep(sec * 1000);
#endif
}

void mySleepMs(int msec) {
#ifdef __linux__
    usleep(msec * 1000);
#elif _WIN32
    Sleep(msec);
#endif
}

uint32_t restoreNumberFromLowBit(uint32_t lowBit, uint8_t lowBitSize, uint32_t lowRange, uint32_t highRange) {
    uint32_t mask = (1u << lowBitSize) - 1;
    uint32_t choice1 = (lowRange & ~mask) + (lowBit & mask);
    uint32_t choice2 = (highRange & ~mask) + (lowBit & mask);

    if (choice1 >= lowRange && choice1 <= highRange)
        return choice1;
    if (choice2 >= lowRange && choice2 <= highRange)
        return choice2;

    throw std::runtime_error("Unable to restore number from range");
}

int timeIt(const std::function<void()>& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count();
}

std::string readFile(const std::string& filename) {
    std::ifstream stringsFile(filename, std::ios::binary);
    if (!stringsFile.good()) {
        throw std::runtime_error("Error reading strings file");
    }

    stringsFile.seekg(0, std::ifstream::end);
    size_t fileSize = stringsFile.tellg();
    stringsFile.seekg(0, std::ifstream::beg);

    char *buf = new char[fileSize];
    stringsFile.read(buf, fileSize);
    std::string strings(buf, fileSize);
    delete[] buf;

    return strings;
}