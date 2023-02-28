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
        28 * 28, 64, 10
    };

    int num_layers = sizeof(layer_sizes) / sizeof(int); // calculate the size of layer_sizes

    try {
        Network network(layer_sizes, num_layers);

        // make last layer activation function, sigmoid:
        network.layers[network.layers.size() - 1]->activation_function = Layer::Function::Sigmoid;
        network.layers[network.layers.size() - 2]->activation_function = Layer::Function::RELU;

        Trainer trainer(network);

        // open test data and labels files
        trainer.training_data.set_test_data_file("/home/naqeeb/MNIST-DNN-Training/training_data/bin/test-images.idx3-ubyte");
        trainer.training_data.set_test_labels_file("/home/naqeeb/MNIST-DNN-Training/training_data/bin/test-labels.idx1-ubyte");

        trainer.training_data.get_test_data();
        trainer.training_data.get_test_labels();

        SPDLOG_INFO("Accuracy: " + to_string(trainer.test_network() * 100) + "%");

        trainer.training_data.set_training_data_file("/home/naqeeb/MNIST-DNN-Training/training_data/bin/train-images.idx3-ubyte");
        trainer.training_data.set_training_labels_file("/home/naqeeb/MNIST-DNN-Training/training_data/bin/train-labels.idx1-ubyte");

        int epoch = 1;
        while (true) {
            SPDLOG_INFO("Training epoch " + epoch);
            trainer.train_epoch();
            SPDLOG_INFO("Accuracy: " + to_string(trainer.test_network() * 100) + "%");
            epoch++;
        }


    } catch (invalid_argument e) {
        SPDLOG_ERROR(e.what());
        return 1;
    } catch (invalid_function_call e) {
        SPDLOG_ERROR(e.what());
        return 1;
    }

    return 0;
}