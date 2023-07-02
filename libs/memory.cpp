#include "memory.h"

#ifdef ENABLE_POOLING
std::array<PreAllocPool, static_cast<int>(HOST_ONLY) + 1> memPools;
#endif