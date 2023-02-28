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
     * @brief Neural network instance this trainer is being created for
     */
    Network* network = NULL;

    // We don't want to reallocate this memory for every record trained, we do it once.

    /**
     * @brief 3-D array storing activation of each layer in network in every training batch.
     */
    float*** activations = NULL;

    /**
     * @brief 2D array containing the error for each layer in network EXCEPT for the input layer (The input layer has
     *        no bias or weights so no weights/biases to train). Used for storing errors that are used for
     *        backpropagation. For every training batch.
     */
    float*** error = NULL;

    /**
     * @brief 3D array containing the cost gradients of each weight in each layer (expect for input, which has no weights or
     *        biases). The first dimension represents the layer, the second dimension represents a neuron in the layer, and
     *        the third dimension are the weights from previous layer to the neuron. We only require one for all batches
     *        because the calculated weight gradient will just be added together and divided by the batch size to obtain
     *        the average weight_gradient for the entire training data batch
     */
    float*** weight_gradient = NULL;

    /**
     * @brief Store a copy of network layer sizes here incase the original network object is deleted.
     *        this is to make sure we can still delete allocated memory to prevent leaks.
     */
    vector<int> layer_sizes;

public:

    float step_size = 0.005f;

    void setNetwork(Network& network) {
        this->network = &network;

        // initialize the activations, error, and weight_gradient array:

        /**
         * TODO: Fix this pointer hell. One idea is to collapse 2-D & 3-D arrays into a single 1-D array
         *       this should also be slightly faster
         *       https://stackoverflow.com/questions/17259877/1d-or-2d-array-whats-faster
        */

        for (int l = 0; l < network.layers.size(); l++) {
            layer_sizes.push_back(network.layers[l]->size);
        }

        activations = new float** [training_data.batch_size];
        for (int b = 0; b < training_data.batch_size; b++) {
            activations[b] = new float* [network.layers.size()];
            for (int x = 0; x < network.layers.size(); x++) {
                activations[b][x] = new float[network.layers[x]->size];
            }
        }

        error = new float** [training_data.batch_size];
        for (int b = 0; b < training_data.batch_size; b++) {
            error[b] = new float* [network.layers.size() - 1];
            for (int x = 0; x < network.layers.size() - 1; x++) {
                error[b][x] = new float[network.layers[x + 1]->size];
            }
        }

        weight_gradient = new float** [network.layers.size() - 1];
        for (int l = 1; l < network.layers.size(); l++) {
            weight_gradient[l - 1] = new float* [network.layers[l - 1]->size];
            for (int x = 0; x < network.layers[l - 1]->size; x++) {
                weight_gradient[l - 1][x] = new float[network.layers[l]->size];
            }
        }
    }

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
        setNetwork(network);
    }

    ~Trainer() {
        for (int b = 0; b < training_data.batch_size; b++) {
            for (int x = 0; x < layer_sizes.size(); x++) {
                delete[] activations[b][x];
            }
            delete[] activations[b];
        }
        delete[] activations;

        for (int b = 0; b < training_data.batch_size; b++) {
            for (int x = 0; x < layer_sizes.size() - 1; x++) {
                delete[] error[b][x];
            }
            delete[] error[b];
        }
        delete[] error;

        for (int l = 1; l < layer_sizes.size(); l++) {
            for (int x = 0; x < layer_sizes[l - 1]; x++) {
                delete[] weight_gradient[l - 1][x];
            }
            delete[] weight_gradient[l - 1];
        }
        delete[] weight_gradient;

        SPDLOG_DEBUG("Deleted trainer");
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

            if (t == 0) {
                // print last layer ouputs
                for (int x = 0; x < layer_sizes[layer_sizes.size() - 1]; x++) {
                    SPDLOG_DEBUG(to_string(x) + ": " + to_string(activations_per_layer[layer_sizes.size() - 1][x]));
                }
            }
        }

        for (int x = 0; x < network->layers.size(); x++) {
            delete[] activations_per_layer[x];
        }
        delete[] activations_per_layer;

        return correct_guesses / (float)training_data.test_data_items_count;
    }

    /**
     * @brief Train the network on one epoch of training data. The neural network is given the training data in batches.
     *        If there are N batches of training data, an epoch occurs when all N batches have been seen by the network once.
     */
    void train_epoch() {
        if (network == NULL) {
            throw invalid_function_call("Trainer does not have any network to train");
        }

        for (int x = 0; x < training_data.total_batch_count; x++) {
            train_next_batch();
        }
    }

    /**
     * @brief Train the network on the next batch of training data. Note that the last batch may be smaller than batch size.
     */
    void train_next_batch() {
        // SPDLOG_INFO("Training batch " + to_string(training_data.current_batch) + "/" +
            // to_string(training_data.total_batch_count));

        training_data.get_next_training_data_batch();
        training_data.get_next_training_labels_batch();

        int batch_size = training_data.batch_size;
        if (training_data.current_batch == training_data.total_batch_count) {
            // on the last batch so batch size will be different,
            batch_size = training_data.training_data_items_count % batch_size;
        }

        // set all values to zero in activations and weight_gradient matrix.

        for (int b = 0; b < training_data.batch_size; b++) {
            for (int l = 0; l < network->layers.size(); l++) {
                for (int x = 0; x < network->layers[l]->size; x++) {
                    activations[b][l][x] = 0;
                }
            }
        }

        for (int l = 1; l < network->layers.size(); l++) {
            for (int x = 0; x < network->layers[l - 1]->size; x++) {
                for (int y = 0; y < network->layers[l]->size; y++) {
                    weight_gradient[l - 1][x][y] = 0;
                }
            }
        }



        for (int x = 0; x < batch_size; x++) {
            train_record(training_data.training_data_batch_buffer[x], training_data.training_labels_batch_buffer[x], x);
        }

        // weight gradients now contains the sum of weight gradients of all training records,
        // dividing each gradient by the batch size gives us the average gradient vector of all
        // training records in the batch. Now we update the weights and biases,

        /**
         * @brief The average weight gradient is dW/dC * (step_size / batch_size)
         *        rather than calculating this everytime, we do it once here.
         */
        float coefficient = step_size / training_data.batch_size;

        // first update weights
        for (int l = 1; l < layer_sizes.size(); l++) {
            for (int x = 0; x < layer_sizes[l - 1]; x++) {
                for (int y = 0; y < layer_sizes[l]; y++) {
                    network->layers[l]->weights[x][y] -= weight_gradient[l - 1][x][y] * coefficient;
                }
            }
        }

        // now update biases, we need to calculate average bias change for each neuron first
        for (int l = 1; l < layer_sizes.size(); l++) {
            for (int x = 0; x < layer_sizes[l]; x++) {
                float average_error = 0;
                for (int b = 0; b < training_data.batch_size; b++) {
                    average_error += error[b][l - 1][x];
                }
                average_error /= training_data.batch_size;
                network->layers[l]->biases[x] -= average_error * step_size;
            }
        }
    }

    /**
     * @brief Train on a single record
     *
     * @param record Array containing input to network
     * @param label Label for this record
     * @param batch_record_index Index in record that this batch is part of
     */
    void train_record(float* record, unsigned char label, int batch_record_index) {
        // load record into input layer
        for (int x = 0; x < network->layers[0]->size; x++) {
            activations[batch_record_index][0][x] = record[x];
        }

        network->propagate_backpropagate(
            activations[batch_record_index], error[batch_record_index], weight_gradient,
            label
        );
    }

};