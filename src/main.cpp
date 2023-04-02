#include <iostream>
#include <string>
#include <vector>

#include <CLI/App.hpp> // CLI library is used for getting command line arguments
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>

#include "logging.cpp" // contains #import <spdlog/spdlog.h> as well as configuration defines
#include "network.cpp"
#include "trainer/trainer.cpp"

using namespace std;

int main(int argc, char *argv[]) {

    /**
     * Set up CLI app and parse arguments into variables
     */
    CLI::App app("Create minimum implementation of backpropagation algorithm to train a DNN as both an educational resource "
                 "and a starting point for implementing a training algorithm.",
                 "MNIST-DNN-Training");

    string training_data_file, training_labels_file, test_data_file, test_labels_file;

    bool verbose = false;
    bool log_accuracy = true;

    app.add_option("--training_data", training_data_file, "Path to training data file")->required();
    app.add_option("--training_labels", training_labels_file, "Path to training labels file")->required();
    app.add_option("--test_data", test_data_file, "Path to test data file")->required();
    app.add_option("--test_labels", test_labels_file, "Path to test labels file")->required();

    app.add_flag("-v,--verbose", verbose, "Print out debug information as well")->default_val(false);

    // We can disable logging for whatever reason by passing the --no-logging flag
    app.add_flag("--no-logging{false}", log_accuracy, "Disable logging by passing the --no-logging flag")->default_val(true);

    CLI11_PARSE(app);

    // initalize and configure spdlog
    logging::initialize(verbose);

    if (!log_accuracy) {
        SPDLOG_INFO("--no-logging flag means logging is disabled.");
    }

    int layer_sizes[] = {28 * 28, 40, 10};

    int num_layers = sizeof(layer_sizes) / sizeof(int); // calculate the size of layer_sizes

    try {
        Network network(layer_sizes, num_layers);

        // make last layer activation function, sigmoid:
        network.layers[network.layers.size() - 1]->activation_function = Layer::Function::Sigmoid;

        Trainer trainer(network);

        // open test data and labels files
        trainer.training_data.set_test_data_file(test_data_file);
        trainer.training_data.set_test_labels_file(test_labels_file);

        trainer.training_data.get_test_data();
        trainer.training_data.get_test_labels();

        trainer.training_data.set_training_data_file(training_data_file);
        trainer.training_data.set_training_labels_file(training_labels_file);

        trainer.train(100, log_accuracy);

    } catch (invalid_argument e) {
        SPDLOG_ERROR(e.what());
        return 1;
    } catch (invalid_function_call e) {
        SPDLOG_ERROR(e.what());
        return 1;
    }

    return 0;
}