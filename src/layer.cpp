#include <vector>
#include "neuron.cpp"

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
     * @brief Construct a new input layer. This should only be the first layer of the Network.
     * 
     * @param size Number of neurons in this layer
     */
    Layer(int size) {

        // no weights or biases for input layer
        weights = NULL;
        biases = NULL;

    }

    /**
     * @brief Construct a new hidden layer
     * 
     * @param size Number of neurons in this layer
     * @param previous_layer_size Number of neurons in previous layer. If this is the first layer (input layer), then this
     *                            is the number of input values.
     */
    Layer(int size, int previous_layer_size) {
        // initialize weight matrix
        weights = new float*[previous_layer_size];
        for(int x=0; x<previous_layer_size; x++) {
            weights[x] = new float[size];
        }
    }

};