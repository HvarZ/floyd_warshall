#include <iostream>
#include <string>
#include <memory>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <unistd.h>
#include "lib/apsp.h"

using namespace std;
using namespace std::chrono;

void usageCommand(string message = "") {
    cerr << message << std::endl;
    cout << "Usage: ./binnary.out [-a <int>] < input_data.txt" << endl;
    cout << "-a Type of algorithm for APSP" << endl
            << "\t0: NAIVE FLOYD WARSHALL" << endl
            << "\t1: CUDA BLOCKED FLOYD WARSHALL" << endl;
    cout << "-h for help" << endl;
    exit(-1);
}

graphAPSPAlgorithm parseCommand(int argc, char** argv) {
    int opt;
    graphAPSPAlgorithm algorithm;
    while ((opt = getopt(argc, argv, "ha:")) != -1) {
        switch (opt) {
        case 'a':
            algorithm = static_cast<graphAPSPAlgorithm>(atoi(optarg));
            break;
        case 'h':
            usageCommand();
            break;
        default:
            usageCommand("Unknown argument !!!");
        }
    }
    if (algorithm > graphAPSPAlgorithm::CUDA_BLOCKED_FW ||
            algorithm < graphAPSPAlgorithm::NAIVE_FW)
        usageCommand("Incorrect value for -a !!!");
    return algorithm;
}


unique_ptr<graphAPSPTopology> readData(int maxValue) {
    int nvertex, nedges;
    int v1, v2, value;
    cin >> nvertex >> nedges;

    unique_ptr<graphAPSPTopology> data;
    data = unique_ptr<graphAPSPTopology>(new graphAPSPTopology(nvertex));
    fill_n(data->pred.get(), nvertex * nvertex, -1);
    fill_n(data->graph.get(), nvertex * nvertex, maxValue);

    for (int i=0; i < nedges; ++i) {
        cin >> v1 >> v2 >> value;
        data->graph[v1 * nvertex + v2] = value;
        data->pred[v1 * nvertex + v2] = v1;
    }

    for (unsigned int i=0; i < nvertex; ++i) {
        data->graph[i * nvertex + i] = 0;
        data->pred[i * nvertex + i] = -1;
    }
    return data;
}

void printDataJson(const unique_ptr<graphAPSPTopology>& graph, int time, int maxValue) {
    ios::sync_with_stdio(false);
    auto printMatrix = [](unique_ptr<int []>& graph, int n, int max) {
        cout << "[";
        for (int i = 0; i < n; ++i) {
            cout << "[";
            for (int j = 0; j < n; ++j) {
                if (max > graph[i * n + j])
                    cout << graph[i * n + j];
                else
                    cout << -1 ;
                if (j != n - 1) cout << ",";
            }
            if (i != n - 1)
                cout << "],\n";
            else
                cout << "]";
        }
        cout << "],\n";
    };

    cout << "{\n    \"graph\":\n";
    printMatrix(graph->graph, graph->nvertex, maxValue);
    cout << "    \"compute_time\": " << time << "\n}";
}

int main(int argc, char **argv) {
    int maxValue = MAX_DISTANCE;
    auto algorithm = parseCommand(argc, argv);
    auto graph = readData(maxValue);

    high_resolution_clock::time_point start = high_resolution_clock::now();
    apsp(graph, algorithm);
    high_resolution_clock::time_point stop = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>( stop - start ).count();
    printDataJson(graph, duration, maxValue);
    return 0;
}
