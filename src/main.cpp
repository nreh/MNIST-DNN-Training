#include <iostream>
#include "network.cpp"

using namespace std;

int main() {
    
    int layer_sizes[] = {
        28 * 28, 15, 15, 10
    };

    int num_layers = sizeof(layer_sizes)/sizeof(int); // calculate the size of layer_sizes

    cout << "creation..." << endl; 
    
    Network* network = new Network( layer_sizes, num_layers );

    network->layers[1].weights[0][0] = 0;

    cout << "Created network with " << num_layers << " layers" << endl;

    delete network;

    // // network->layers[1].weights[0][0] = 0;

    return 0;
}