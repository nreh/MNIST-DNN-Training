#include <vector>
#include <random>

#include "config.h"
#include "neuron.cpp"
#include "math_functions.cpp"

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
     * @brief Number of neurons in previous layer. A value of zero indicates that this layer is the input layer.
     */
    int previous_layer_size = 0;

    /**
     * @brief Number of neurons in this layer. 
     */
    int size = 0;

    /**
     * @brief Construct a new input layer. This should only be the first layer of the Network.
     * 
     * @param size Number of neurons in this layer.
     */
    Layer(int size) {

        // no weights or biases for input layer
        weights = NULL;
        biases = NULL;

    }

    /**
     * @brief Construct a new hidden layer.
     * 
     * @param size Number of neurons in this layer.
     * @param previous_layer_size Number of neurons in previous layer. If this is the first layer (input layer), then this
     *                            is the number of input values.
     */
    Layer(int size, int previous_layer_size) {

        this->size = size;
        this->previous_layer_size = previous_layer_size;

        // initialize weight matrix

        weights = new float*[previous_layer_size];
        for(int x=0; x<previous_layer_size; x++) {
            weights[x] = new float[size];
        }

        // randomly initialize weights on a normal distribution

        default_random_engine engine;
        normal_distribution<float> distr(INIT_NORMAL_MEAN, INIT_NORMAL_STDDEV);

        for(int x=0; x<previous_layer_size; x++) {
            for(int y=0; y<size; y++) {
                weights[x][y] = distr(engine);
            }
        }

        // randomly initialize biases

        biases = new float[size];

        for(int x=0; x<size; x++) {
            biases[x] = distr(engine);
        }

    }

    /**
     * @brief Propagate data through layer and output result to a destination array.
     * 
     * @param in Input values to propagate through. Size is equal to previous_layer_size.
     * @param out Destination array to output result to. Size is equal to this layer size.
     */
    void propagate(float* in, float* out) {

        // matrix multiplication of in and weight-matrix

        out = new float[size]();

        for(int x=0; x<previous_layer_size; x++) {
            dotProduct(in[x], weights[x], out, size);
        }

    }

    ~Layer() { }

};