#ifndef _APSP_
#define _APSP_

#include <memory>

#define MAX_DISTANCE 1 << 30 - 1

typedef enum {
    NAIVE_FW = 0,
    CUDA_BLOCKED_FW = 1
} graphAPSPAlgorithm;

struct graphAPSPTopology {
    unsigned int nvertex;
    std::unique_ptr<int[]> pred;
    std::unique_ptr<int[]> graph;

    graphAPSPTopology(int nvertex): nvertex(nvertex) {
        int size = nvertex * nvertex;
        pred = std::unique_ptr<int[]>(new int[size]);
        graph = std::unique_ptr<int[]>(new int[size]);
    }
};

void apsp(const std::unique_ptr<graphAPSPTopology>& data, graphAPSPAlgorithm algorithm);

#endif