#include <iostream>

#include "network.cpp"
#include "logging.cpp" // contains #import <spdlog/spdlog.h> as well as configuration defines

using namespace std;

int main(int argc, char* argv[]) {
    // initalize and configure spdlog
    logging::initialize(argv);

    int layer_sizes[] = {
        28 * 28, 15, 15, 10
    };

    int num_layers = sizeof(layer_sizes) / sizeof(int); // calculate the size of layer_sizes

    Network* network;

    try {
        network = new Network(layer_sizes, num_layers);
    } catch (runtime_error e) {
        SPDLOG_ERROR(e.what());
        return 1;
    }

    network->layers[1].weights[0][0] = 0;

    delete network;

    // // network->layers[1].weights[0][0] = 0;

    return 0;
}