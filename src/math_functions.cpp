/**
 * Math functions used throughout training and testing
 */

class ActivationFunctions {
  public:
    // σ(x) = x when x >= 0  &  σ(x) = 0 when x < 0
    static float RELU(float x) {
        if (x < 0) {
            return 0;
        } else {
            return x;
        }
    }

    static float sigmoid(float x) {
        // we use fast sigmoid function:
        return x / (1 + abs(x));
    }
};

class ActivationFunctionGradients {
  public:
    // σ′(x) = 1 when x >= 0  &  σ′(x) = 0 when x < 0
    static float RELU_gradient(float x) {
        if (x < 0) {
            return 0;
        } else {
            return 1;
        }
    }

    static float sigmoid_gradient(float x) {
        // we use fast sigmoid function:
        return 1 / pow(abs(x) + 1, 2);
    }
};

void dot_product(float a, float *b, float *c, int length) {
    for (int x = 0; x < length; x++) {
        c[x] += a * b[x];
    }
}