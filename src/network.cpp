#pragma once

#include <iostream>
#include <vector>
#include <stdexcept> // for exceptions
#include <string>

#include "logging.h" // for logging

#include "layer.cpp"

using namespace std;

/**
 * A neural network with layers and neurons in each layer. Has functions used to execute training iterations.
 */
class Network {
public:

    /**
     * List of all layers in network
     */
    vector<Layer*> layers;

    /**
     * @brief Construct a new Neural Network
     *
     * @param layer_sizes An array containing size of each layer in network
     * @param num_layers Number of layers in network (AKA the size of layer_sizes array)
     */
    Network(int layer_sizes[], int num_layers) {
        /**
         * Error checking
         */
        if (num_layers < 2) {
            throw std::invalid_argument("Network must contain at least 2 layers");
        }

        // first create the input layer,
        layers.push_back(new Layer(layer_sizes[0]));

        // now create hidden and ouput layers,
        for (int x = 1; x < num_layers; x++) {
            layers.push_back(new Layer(layer_sizes[x], layer_sizes[x - 1], x));
        }

        SPDLOG_DEBUG("Created network with {0} layers", num_layers);
    }

    /*------------------------------------------- Training Functions -------------------------------------------*/

    /**
     * @brief Propagate input through neural network.
     *
     * @param activations 2D array containing the activations of each layer in network. the first array will contain the
     *                    input to the network and last array will contain the ouput values once input propagates. After
     *                    propagation, all activations are stored in the activations array.
     */
    void propagate(float** activations) {
        /**
         *?                         ==================================================
         *?                                       🛈 How propagation works
         *?                         ==================================================
         *
         * Take two adjacent layers called A & B. The first neuron in layer A is called A0.
         *
         * To calculate the activation of a single neuron, let's say B0:
         *  - Multiply activation of A0 with weight of neuron connection from A0 to B0 & add bias of B0
         *  - Do the same for the rest of the neurons in layer A and add them up, let's call the resulting value: z
         *  - So, z = Σ(activations_A * weights_A_B + biases_B)
         *  - The activation of B0 will be f(z) where f is the activation function of B0 (ex. RELU, Sigmoid)
         *
        */

        for (int x = 1; x < layers.size(); x++) {
            layers[x]->propagate(activations[x - 1], activations[x]);
        }
    }

    // Clean up

    ~Network() {
        for (int x = 0; x < layers.size();x++) {
            delete layers[x];
        }

        SPDLOG_DEBUG("Deleted network");
    }

};