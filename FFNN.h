#ifndef FORWARD_NETWORK_H
#define FORWARD_NETWORK_H

#define nodesIndexFFNN (layer - 1)

struct Node {
    float bias;
    float* weights;
};

struct ForwardLog {
    // stores inputs into a node
    // info needed for backpropagation
    float* nodeInputs;
};

struct NodeGradient {
    // stores results from back propagation
    // partial derivative with respect to cost function
    float dBias;
    float* dWeights;
};

// feed forward neural network
struct FFNN {
    int numLayers; // including inputs, hidden layers, and outputs
    int* layerSizes;
    struct Node** nodes;

    float** forwardVals; // outputs from each node after forward pass
    struct ForwardLog** forwardLog; // holds required info for backpropagation
};

struct FFNN* alloc(int numLayers, int* layerSizes);

void randomize(struct FFNN* ffnn);

void print(struct FFNN* ffnn);

void setInput(struct FFNN* ffnn, float* inputs);

void forwardPass(struct FFNN* ffnn);

float* getOutput(struct FFNN* ffnn);

float quadraticCost(float* prediction, float* actual, int size);

#endif