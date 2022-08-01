#include <iostream>
#include "network.cpp"

using namespace std;

int main() {
    
    int layer_sizes[] = {
        28 * 28, 15, 15, 10
    };

    int num_layers = sizeof(layer_sizes)/sizeof(int); // calculate the size of layer_sizes

    Network* network = new Network( layer_sizes, num_layers );

    cout << "Created network with " << num_layers << " layers" << endl;    

    return 0;
}