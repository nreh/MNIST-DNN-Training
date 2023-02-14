#pragma once

#include <string>
#include <fstream>

#include "training_data.cpp"

#include "../config.h"
#include "../network.cpp"
#include "../logging.h"

class Trainer {
public:

    /**
     * @brief Neural network instance this trainer is being created for
     */
    Network* network = NULL;

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
     * @brief Training data is stored here
     */
    TrainingData training_data;


};