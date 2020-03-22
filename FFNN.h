#ifndef FORWARD_NETWORK_H
#define FORWARD_NETWORK_H

// ====Helper defines for indexing notation====
// L is layer index, J is neuron index in layer, K is neuron index in (L-1)th layer.

// Weight from K output to J input
#define W(L, J, K) (ffnn->nodes[L][J].weights[K])

// Bias
#define B(L, J) (ffnn->nodes[L][J].bias)

// Output computed by neuron from sum and activation function
#define A(L, J) (ffnn->forwardVals[L][J])

struct Node {
    float bias;
    float* weights;
};

struct NodeGradient {
    float dBias;
    float* dWeights;
};

// feed forward neural network
struct FFNN {
    int numLayers; // including inputs, hidden layers, and outputs
    int* layerSizes;
    struct Node** nodes;

    float** forwardVals; // outputs from each node after forward pass

    // boolean indicating usage of softmax on output nodes
    // and cross entropy for cost
    int categorical;
};

struct FFNN* alloc(int numLayers, int* layerSizes);

void setCategorical(struct FFNN* ffnn);

void setRegressional(struct FFNN* ffnn);

void randomize(struct FFNN* ffnn);

void setNetwork(struct FFNN* ffnn, float** vals);

void print(struct FFNN* ffnn);

void setInput(struct FFNN* ffnn, float* inputs);

void forwardPass(struct FFNN* ffnn);

float* getOutput(struct FFNN* ffnn);

float quadraticCost(float* prediction, float* actual, int size);

struct NodeGradient** backwardPass(struct FFNN* ffnn, float* actual);

void applyGradient(struct FFNN* ffnn, struct NodeGradient** gradient, float learningRate);

void SGD(struct FFNN* ffnn,
                    float** inputs, 
                    float** outputs, 
                    int trainingSetSize, 
                    float learningRate,
                    int epochs);

#endif