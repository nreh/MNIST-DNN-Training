#include <iostream>
#include <vector>

#include "logging.cpp" // contains #import <spdlog/spdlog.h> as well as configuration defines
#include "network.cpp"
#include "trainer/trainer.cpp"

using namespace std;

int main(int argc, char *argv[]) {
    // initalize and configure spdlog
    logging::initialize(argv);

    int layer_sizes[] = {28 * 28, 40, 10};

    int num_layers = sizeof(layer_sizes) / sizeof(int); // calculate the size of layer_sizes

    try {
        Network network(layer_sizes, num_layers);

        // make last layer activation function, sigmoid:
        network.layers[network.layers.size() - 1]->activation_function = Layer::Function::Sigmoid;

        Trainer trainer(network);

        // open test data and labels files
        trainer.training_data.set_test_data_file("../training_data/bin/test-images.idx3-ubyte");
        trainer.training_data.set_test_labels_file("../training_data/bin/test-labels.idx1-ubyte");

        trainer.training_data.get_test_data();
        trainer.training_data.get_test_labels();

        trainer.training_data.set_training_data_file("../training_data/bin/train-images.idx3-ubyte");
        trainer.training_data.set_training_labels_file("../training_data/bin/train-labels.idx1-ubyte");

        trainer.train(100);

    } catch (invalid_argument e) {
        SPDLOG_ERROR(e.what());
        return 1;
    } catch (invalid_function_call e) {
        SPDLOG_ERROR(e.what());
        return 1;
    }

    return 0;
}