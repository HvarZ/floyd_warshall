#include <functional>
#include "apsp.h"
#include "cuda/cuda_apsp.cuh"

static
void naiveFW(const std::unique_ptr<graphAPSPTopology>& data) {
    int newPath = 0;
    int n = data->nvertex;

    for(int u=0; u < n; ++u) {
        for(int v1=0; v1 < n; ++v1) {
            for(int v2=0; v2 < n; ++v2) {
                newPath = data->graph[v1 * n + u] + data->graph[u * n + v2];
                if (data->graph[v1 * n + v2] > newPath) {
                    data->graph[v1 * n + v2] = newPath;
                    data->pred[v1 * n + v2] = data->pred[u * n + v2];
                }
            }
        }
    }
}

void apsp(const std::unique_ptr<graphAPSPTopology>& data, graphAPSPAlgorithm algorithm) {
    std::function<void(const std::unique_ptr<graphAPSPTopology>&)> algorithms[] = {
            naiveFW,
            cudaBlockedFW
    };
    algorithms[algorithm](data);
}
