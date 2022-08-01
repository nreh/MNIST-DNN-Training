/**
 * Math functions used throughout training and testing
 */

class ActivationFunctions {

    static float RELU(float x) {
        if (x < 0) {
            return 0;
        } else {
            return x;
        }
    }

    static float sigmoid(float x) {
        //todo: implement
    }

};

void dotProduct(float a, float* b, float* c, int length) {
    for(int x=0; x<length; x++) {
        c[x] += a * b[x];
    }
}