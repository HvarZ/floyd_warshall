#ifndef _CUDA_APSP_
#define _CUDA_APSP_

#include "../apsp.h"

#define BLOCK_SIZE 16

void cudaBlockedFW(const std::unique_ptr<graphAPSPTopology>& dataHost);

#endif /* _APSP_ */
