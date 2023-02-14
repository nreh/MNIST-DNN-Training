#include <iostream>
#include <vector>

#include "network.cpp"
#include "logging.cpp" // contains #import <spdlog/spdlog.h> as well as configuration defines
#include "trainer.cpp"

using namespace std;

int main(int argc, char* argv[]) {
    // initalize and configure spdlog
    logging::initialize(argv);

    int layer_sizes[] = {
        28 * 28, 15, 15, 10
    };

    int num_layers = sizeof(layer_sizes) / sizeof(int); // calculate the size of layer_sizes

    vector<Trainer> trainers;

    try {
        Network network(layer_sizes, num_layers);

        Trainer trainer(network);
        trainers.push_back(trainer);
    } catch (invalid_argument e) {
        SPDLOG_ERROR(e.what());
        return 1;
    }

    return 0;
}