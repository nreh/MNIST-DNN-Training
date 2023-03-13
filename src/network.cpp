#pragma once

#include <iostream>
#include <stdexcept> // for exceptions
#include <string>
#include <vector>

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
    vector<Layer *> layers;

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

        // create a random engine
        default_random_engine engine(time(0));

        // now create hidden and ouput layers,
        for (int x = 1; x < num_layers; x++) {
            layers.push_back(new Layer(layer_sizes[x], layer_sizes[x - 1], x, engine));
        }

        SPDLOG_DEBUG("Created network with {0} layers", num_layers);
    }

    /*------------------------------------------- Training Functions -------------------------------------------*/

    /**
     * @brief Propagate input through neural network.
     *
     * @param activations 2D array containing the activations of each layer in network. the first array will contain the
     *                    input to the network and last array will contain the ouput values once input propagates. After
     *                    propagation, all activations are stored in the activations array. Should already be initialzed
     *                    and have hidden/output layer activations initialized to 0.
     */
    void propagate(float **activations) {
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
         *  - The activation of B0 will be σ(z) where σ is the activation function of B0 (ex. RELU, Sigmoid)
         *
         */

        for (int x = 1; x < layers.size(); x++) {
            layers[x]->propagate(activations[x - 1], activations[x]);
        }
    }

    /**
     * @brief Propagate input through network and then back propagate calculating weight and bias gradients.
     *
     * @param activations 2D array containing the activations of each layer in network. the first array will contain the
     *                    input to the network and last array will contain the ouput values once input propagates. After
     *                    propagation, all activations are stored in the activations array. Should already be initialzed
     *                    and have hidden/output layer activations initialized to 0.
     *
     * @param error 2D array containing the error for each layer in network EXCEPT for the input layer (The input layer has
     *              no bias or weights so no weights/biases to train). Used for storing errors that are used for
     *              backpropagation. Should already be initialized.
     *
     * @param weight_gradient 3D array containing the cost gradients of each weight in each layer. The first dimension
     *                        represents the layer, the second dimension represents a neuron in the layer, and the third
     *                        dimension are the weights from previous layer to the neuron. Should already be initialized
     *                        with all values set to 0. (This is so that average weight gradient can be calculated from
     *                        multiple records). Because the bias gradient is just equal to the error, we don't need a bias
     *                        gradient array.
     *
     * @param label The label for the training data. In this network, this will be the neuron with highest activation in the
     *              output layer.
     *
     * @param average_loss The loss of the network on each training record are added to this variable. Divide it by the
     *                     total number of training records/batch-size to obtain average loss for all training data/batch.
     */
    void propagate_backpropagate(float **activations, float **error, float ***weight_gradient, unsigned char label) {
        // first propagate input through all layers, while also calculating the gradient of the activation function,
        for (int l = 1; l < layers.size(); l++) {
            // The the gradient of the activation function will be stored in the error array. Later, we'll multiply it
            // with gradients/weights (depending on the layer) to calculate the error for that layer

            // note that we are doing error[l-1], this is because the error array has the length equal to the number of
            // layers in the network - 1 as the input layer has no error that needs to be calculated.
            layers[l]->propagate_backpropagate(activations[l - 1], activations[l], error[l - 1]);
        }

        // calculate the error for the output layer,
        for (int x = 0; x < layers[layers.size() - 1]->size; x++) {
            /**
             * We're using the quadratic cost function, so the derivative of the cost function C for a given neuron with
             * activation a is:
             *
             *      dC/da = a − y
             *
             * Where y is the desired activation of the neuron.
             */
            // Once again, we subtract layers.size by 2 because the length of the error array is the number of layers in the
            // network - 1 as the input layer has no error that needs to be calculated.
            if (x == label) {
                error[layers.size() - 2][x] *= activations[layers.size() - 1][x] - 1; // we wanted an activation of 1
            } else {
                error[layers.size() - 2][x] *= activations[layers.size() - 1][x] - 0; // we wanted an activation of 0
            }
        }

        // now backpropagate
        for (int l = 1; l < layers.size() - 1; l++) {
            for (int x = 0; x < layers[l]->size; x++) {
                // calculate dot-product between this & next layer's weights and the error of the next layer,
                float dot_product = 0;
                for (int y = 0; y < layers[l + 1]->size; y++) {
                    dot_product += layers[l + 1]->weights[x][y] * error[l][y];
                }
                error[l - 1][x] *= dot_product;
            }
        }

        // now calculate how much weights change
        for (int l = 1; l < layers.size(); l++) {
            for (int x = 0; x < layers[l - 1]->size; x++) {
                for (int y = 0; y < layers[l]->size; y++) {
                    // we subtract 1 from l because like the error matrix, the weight gradient matrix doesn't include the
                    // input layer as there are no weights to train.
                    weight_gradient[l - 1][x][y] += activations[l - 1][x] * error[l - 1][y];
                }
            }
        }
    }

    // Clean up

    ~Network() {
        for (int x = 0; x < layers.size(); x++) {
            delete layers[x];
        }

        SPDLOG_DEBUG("Deleted network");
    }
};