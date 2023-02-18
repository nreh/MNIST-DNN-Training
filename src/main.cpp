#include <iostream>
#include <vector>

#include "network.cpp"
#include "logging.cpp" // contains #import <spdlog/spdlog.h> as well as configuration defines
#include "trainer/trainer.cpp"

using namespace std;

int main(int argc, char* argv[]) {
    // initalize and configure spdlog
    logging::initialize(argv);

    int layer_sizes[] = {
        28 * 28, 15, 15, 10
    };

    int num_layers = sizeof(layer_sizes) / sizeof(int); // calculate the size of layer_sizes



    try {

        vector<Trainer> trainers;

        Network network(layer_sizes, num_layers);

        Trainer trainer(network);
        trainers.push_back(move(trainer));

        // open test data and labels files
        trainer.training_data.set_test_data_file("/home/naqeeb/MNIST-DNN-Training/training_data/bin/test-images.idx3-ubyte");
        trainer.training_data.set_test_labels_file("/home/naqeeb/MNIST-DNN-Training/training_data/bin/test-labels.idx1-ubyte");

        trainer.training_data.get_test_data();
        trainer.training_data.get_test_labels();

        SPDLOG_INFO("Accuracy: " + to_string(trainer.test_network() * 100) + "%");


    } catch (invalid_argument e) {
        SPDLOG_ERROR(e.what());
        return 1;
    } catch (invalid_function_call e) {
        SPDLOG_ERROR(e.what());
        return 1;
    }

    return 0;
}