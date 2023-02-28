#include <vector>
#include <random>

#include "config.h"
#include "neuron.cpp"
#include "math_functions.cpp"
#include "exceptions.h"

using namespace std;

/**
 * @brief A layer in the neural network.
 */
class Layer {
public:

    /**
     * Contains biases for every neuron in the layer.
     */
    float* biases;

    /**
     * 2-D array containing weights for each neuron in the previous layer to each neuron in the current layer.
     *
     * * This is the Weight-Matrix *
     */
    float** weights;

    /**
     * @brief Number of neurons in previous layer. If this is the first layer (input layer), then this
    *         is the number of input values.
     */
    int previous_layer_size = 0;

    /**
     * @brief Number of neurons in this layer.
     */
    int size = 0;

    /**
     * @brief Index in the neural network that this layer is in.
     */
    int layer_index = 0;

    enum Function {
        RELU,
        Sigmoid
    };

    /**
     * @brief Activation function used by this layer
     */
    Function activation_function = Function::RELU;

    /**
     * @brief Construct a new input layer. This should only be the first layer of the Network.
     *
     * @param size Number of neurons in this layer.
     */
    Layer(int size) {
        this->size = size;

        // no weights or biases for input layer
        weights = NULL;
        biases = NULL;

        SPDLOG_DEBUG("Created input layer of size " + to_string(size));
    }

    /**
     * @brief Construct a new hidden layer.
     *
     * @param size Number of neurons in this layer.
     * @param previous_layer_size Number of neurons in previous layer. If this is the first layer (input layer), then this
     *                            is the number of input values.
     * @param layer_index Index in the neural network that this layer is in.
     */
    Layer(int size, int previous_layer_size, int layer_index) {

        this->size = size;
        this->previous_layer_size = previous_layer_size;
        this->layer_index = layer_index;

        // initialize weight matrix

        weights = new float* [previous_layer_size];
        for (int x = 0; x < previous_layer_size; x++) {
            weights[x] = new float[size];
        }

        // randomly initialize weights on a normal distribution

        default_random_engine engine(time(0));
        normal_distribution<float> distr(INIT_NORMAL_MEAN, INIT_NORMAL_STDDEV);

        for (int x = 0; x < previous_layer_size; x++) {
            for (int y = 0; y < size; y++) {
                weights[x][y] = distr(engine);
            }
        }

        // randomly initialize biases

        biases = new float[size];

        for (int x = 0; x < size; x++) {
            biases[x] = distr(engine);
        }

        SPDLOG_DEBUG("Created hidden/output layer of size " + to_string(size));
    }

    /**
     * @brief Propagate data through layer and output result to a destination array.
     *
     * @param in Input values to propagate through (Previous layer activations). Size is equal to previous_layer_size.
     * @param out Destination array to output final layer activations to (σ(z)). Size is equal to this layer size. Should
     *            already be allocated and have input layer values set as well as 0s for every other layer.
     */
    void propagate(float* in, float* out) {

        // matrix multiplication of in and weight-matrix

        // the input layer cannot have propagate called on it
        if (layer_index == 0) {
            throw invalid_function_call("The propagate function cannot be called on the input layer.");
        }

        for (int x = 0; x < previous_layer_size; x++) {
            dot_product(in[x], weights[x], out, size);
        }

        // apply activation function,


        if (activation_function == RELU) {
            for (int x = 0; x < size; x++) {
                out[x] += biases[x];
                out[x] = ActivationFunctions::RELU(out[x]);
            }
        } else if (activation_function == Sigmoid) {
            for (int x = 0; x < size; x++) {
                out[x] += biases[x];
                out[x] = ActivationFunctions::sigmoid(out[x]);
            }
        }

    }

    /**
     * @brief Propagate input through layer and store activations in output array while calculating output gradient
     *        that is used for calculating the error used for backpropagation.
     *
     * @param in Input values to propagate through (Previous layer activations). Size is equal to previous_layer_size.
     *
     * @param out Destination array to output final layer activations to (σ(z)). Size is equal to this layer size. Should
     *            already be allocated and have input layer values set as well as 0s for every other layer.
     *
     * @param gradient_out Destination array to write the gradient of activation function given input from previous layer
     *                     (σ′(z)). Size is equal to this layer size. This is later used when backpropagating to calculate
     *                     the error for each layer/neuron in the network.
     */
    void propagate_backpropagate(float* in, float* out, float* gradient_out) {

        // matrix multiplication of in and weight-matrix

        // the input layer cannot have propagate called on it
        if (layer_index == 0) {
            throw invalid_function_call("The propagate function cannot be called on the input layer.");
        }

        for (int x = 0; x < previous_layer_size; x++) {
            dot_product(in[x], weights[x], out, size);
        }

        // for each neuron in this layer, copy the weight activations from previous layer + bias into the weight & bias
        // gradient
        for (int x = 0; x < previous_layer_size; x++) {
            dot_product(in[x], weights[x], out, size);
        }

        // Before overwriting out[] array by running them through the activation funcition, we calculate and write the
        // activation funciton gradient to out_gradient[]. (out_gradient = σ′(z))
        if (activation_function == RELU) {
            for (int x = 0; x < size; x++) {
                out[x] += biases[x];
                gradient_out[x] = ActivationFunctionGradients::RELU_gradient(out[x]);
            }
        } else if (activation_function == Sigmoid) {
            for (int x = 0; x < size; x++) {
                out[x] += biases[x];
                gradient_out[x] = ActivationFunctionGradients::sigmoid_gradient(out[x]);
            }
        }

        // apply activation function,
        if (activation_function == RELU) {
            for (int x = 0; x < size; x++) {
                out[x] = ActivationFunctions::RELU(out[x]);
            }
        } else if (activation_function == Sigmoid) {
            for (int x = 0; x < size; x++) {
                out[x] = ActivationFunctions::sigmoid(out[x]);
            }
        }
    }

    ~Layer() {

        // the input layer has no biases or weights to delete
        if (layer_index > 0) {
            SPDLOG_DEBUG("Deleting weights/biases for layer " + to_string(layer_index));

            delete[] biases;

            for (int x = 0; x < previous_layer_size; x++) {
                delete[] weights[x];
            }
            delete[] weights;
        }

    }

};