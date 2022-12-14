#include <iostream>
#include <vector>
#include <stdexcept> // for exceptions
#include <string>

#include "layer.cpp"

using namespace std;

using namespace std;

/**
 * A neural network with layers and neurons in each layer. Has functions used to execute training iterations. 
 */
class Network {
public:
    
    /**
     * List of all layers in network
     */
    vector<Layer> layers;

    /**
     * @brief Construct a new Neural Network
     * 
     * @param layer_sizes An array containing size of layers in network
     * @param num_layers Number of layers in network (AKA the size of layer_sizes array)
     */
    Network(int layer_sizes[], int num_layers) {

        /**
         * Error checking 
         */
        if (num_layers < 2) {
            cout << "Network must contain at least 2 layers" << endl;
        }

        // first create the input layer,
        layers.push_back( Layer(layer_sizes[0]) );

        // now create hidden and ouput layers,
        for(int x=1; x<num_layers; x++) {
            layers.push_back( Layer(layer_sizes[x], layer_sizes[x-1]) );
        }

    }

    /*------------------------------------------- Training Functions -------------------------------------------*/

    /**
     * @brief Propagate input through neural network. 
     * 
     * @param input Input values to network as an array of floats.
     * @param input_size Size of input array.
     * @return float* array of floats with length equal to the size of the output layer.
     */
    float* propagate(float* input, int input_size) {
        return NULL;
    }


    // Clean up

    ~Network() {
        layers.clear();
        cout << "Deleted Network" << endl;
    }    
    
};