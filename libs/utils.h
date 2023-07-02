#ifndef EMBEDLIC_UTILS_H
#define EMBEDLIC_UTILS_H

#include <string>
#include <functional>
#include <chrono>
#include <fstream>
#include <cstdio>
#include <cctype>
#include <memory>

#ifdef __linux__
#include <unistd.h>
#elif _WIN32
#include <windows.h>
#endif

void mySleep(int sec);
void mySleepMs(int msec);
uint32_t restoreNumberFromLowBit(uint32_t lowBit, uint8_t lowBitSize, uint32_t lowRange, uint32_t highRange);

int timeIt(const std::function<void()>& func);
std::string readFile(const std::string& filename);

template<typename ... Args>
std::string stringFormat(const std::string& format, Args ... args)
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ) { throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    auto buf = std::make_unique<char[]>( size );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}


#endif //EMBEDLIC_UTILS_H