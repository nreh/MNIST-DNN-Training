#include <iostream>

#include <spdlog/spdlog.h>

#include "network.cpp"
#include "logging.cpp"

using namespace std;

int main(int argc, char* argv[]) {
    logging::initialize(argv);

    spdlog::info("Welcome to spdlog!");
    spdlog::error("Some error message with arg: {}", 1);

    spdlog::warn("Easy padding in numbers like {:08d}", 12);
    spdlog::critical("Support for int: {0:d};  hex: {0:x};  oct: {0:o}; bin: {0:b}", 42);
    spdlog::info("Support for floats {:03.2f}", 1.23456);
    spdlog::info("Positional args are {1} {0}..", "too", "supported");
    spdlog::info("{:<30}", "left aligned");

    int layer_sizes[] = {
        28 * 28, 15, 15, 10
    };

    int num_layers = sizeof(layer_sizes) / sizeof(int); // calculate the size of layer_sizes

    cout << "creation..." << endl;

    Network* network = new Network(layer_sizes, num_layers);

    network->layers[1].weights[0][0] = 0;

    cout << "Created network with " << num_layers << " layers" << endl;

    delete network;

    // // network->layers[1].weights[0][0] = 0;

    return 0;
}