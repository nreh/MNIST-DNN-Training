#pragma once

#include <string>
#include <fstream>

#include "training_data.cpp"

#include "../config.h"
#include "../network.cpp"
#include "../logging.h"
#include "../exceptions.h"

class Trainer {
private:
    /**
     * @brief Training batch size - default is 100
     */
    int batch_size = 100;

    /**
     * @brief Neural network instance this trainer is being created for
     */
    Network* network = NULL;

public:

    /**
     * @brief Create a new trainer object
     */
    Trainer() {}

    /**
     * @brief Create a new trainer object
     *
     * @param network Network trainer is being created for
     */
    Trainer(Network& network) {
        this->network = &network;
    }

    void setNetwork(Network& network) {
        this->network = &network;
    }

    /**
     *?                             ==================================================
     *?                                               🛈 Training Data
     *?                             ==================================================
     *
     * Training data comprises of two files, input and labels stored in CSV files.
     *
     * The input CSV file contains the inputs to the neural network and each line contains N comma separated floats where N
     * is equal to the number of neurons in the input layer. One record for per line.
     *
     * For example, if training on the MNIST database, one record could be a list of 784 floats (28px * 28px = 784 px²)
     * representing the number 3.
     *
     * The label CSV files contains the correct output layer activations for the given input. Each line contains N comma
     * separated floats where N is the number of neurons in the output layer.
     *
     * So for our MNIST database example, the input data representing a 3 would have the following training label,
     *
     *      0,0,0,1,0,0,0,0,0,0
     *
     * Each of the 10 neurons in the output layer corresponding to a digit 0-9, with the 4th one representing the number 3.
     *
     * Test data is formatted the same as training data and is used for determining how accurate the neural network is on
     * data not in the training set. If the neural network performs well on training data and unwell on test data, that
     * could mean we are over fitting the model.
    */

    /**
     * @brief Instance of TrainingData object used for storing and reading from training data files
     */
    TrainingData training_data;

    /**
     * @brief Train next batch of training data
     */
    void train_batch() {
        if (network == NULL) {
            throw invalid_function_call("Trainer does not have any network to train");
        }

        // read next batch of training data from file
        // training_data.get_next_training_data_batch(training_data_batch_buffer, batch_size);

        //todo: implement backpropagation
    }

    /**
     * @brief Propagate training data through network and return accuracy
     *
     * @return Accuracy (ex, 0.45 is 45% accuracy)
     */
    float test_network() {

        SPDLOG_INFO("Testing neural network on " + to_string(training_data.test_data_items_count) + " test records...");

        if (network == NULL) {
            throw invalid_function_call("Trainer does not have any network to test on");
        }

        // 2-d array that will store our activations,
        float** activations_per_layer = new float* [network->layers.size()];

        for (int x = 0; x < network->layers.size(); x++) {
            activations_per_layer[x] = new float[network->layers[x]->size];
        }

        int correct_guesses = 0; // number of test input that resulted in correct guesses

        // for each test item
        for (int t = 0; t < training_data.test_data_items_count; t++) {
            // first layer activations should be test input,
            for (int x = 0; x < network->layers[0]->size; x++) {
                activations_per_layer[0][x] = training_data.test_data_buffer[t][x];
            }

            // initialize hidden & output layer activations to zero
            for (int x = 1; x < network->layers.size(); x++) {
                for (int y = 0; y < network->layers[x]->size; y++) {
                    activations_per_layer[x][y] = 0;
                }
            }

            network->propagate(activations_per_layer);

            // find index of neuron in output layer that has the highest activation,
            const float* last_layer_activations = activations_per_layer[network->layers.size() - 1];

            int index_of_highest_activation = 0;
            float highest_activation = last_layer_activations[0];

            // for each neuron in output layer...
            for (int n = 1; n < network->layers[network->layers.size() - 1]->size; n++) {
                if (last_layer_activations[n] > highest_activation) {
                    highest_activation = last_layer_activations[n];
                    index_of_highest_activation = n;
                }
            }

            // if highest index equals test label, this guess was a success
            if (training_data.test_labels_buffer[t] == index_of_highest_activation) {
                // correct guess
                correct_guesses++;
            }
        }

        for (int x = 0; x < network->layers.size(); x++) {
            delete[] activations_per_layer[x];
        }
        delete[] activations_per_layer;

        return correct_guesses / (float)training_data.test_data_items_count;
    }

};