#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "FFNN.h"

// layerSizes is shallow copied, but not modified
struct FFNN* alloc(int numLayers, int* layerSizes) {
    struct FFNN *ffnn = malloc(sizeof(*ffnn));
    ffnn->numLayers = numLayers;

    ffnn->layerSizes = malloc(sizeof(*ffnn->layerSizes) * numLayers);
    ffnn->layerSizes = layerSizes;

    ffnn->nodes = malloc(sizeof(*ffnn->nodes) * (numLayers - 1));
    // First layer only contains inputs, not nodes.
    // Leaving empty first layer to allow easy indexing
    ffnn->nodes -= 1;

    for (int l = 1; l < numLayers; ++l) {
        int numNodes = ffnn->layerSizes[l];
        int weightsPerNode = ffnn->layerSizes[l - 1];
        ffnn->nodes[l] = malloc(sizeof(**ffnn->nodes) * numNodes);
        for (int j = 0; j < numNodes; ++j) {
            ffnn->nodes[l][j].weights = malloc(sizeof(*(**ffnn->nodes).weights) * weightsPerNode);
        }
    }

    ffnn->forwardVals = malloc(sizeof(*ffnn->forwardVals) * numLayers);
    for (int l = 0; l < numLayers; ++l) {
        ffnn->forwardVals[l] = malloc(sizeof(**ffnn->forwardVals) * ffnn->layerSizes[l]);
    }

    return ffnn;
}

// ranomizes weights and bias in range [0,1)
void randomize(struct FFNN* ffnn) {
    for (int l = 1; l < ffnn->numLayers; ++l) {
        float scaleDown = ffnn->layerSizes[l - 1];
        for (int j = 0; j < ffnn->layerSizes[l]; ++j) {
            B(l, j) = ((float)rand() / RAND_MAX) / scaleDown;
            for (int k = 0; k < ffnn->layerSizes[l - 1]; ++k) {
                W(l, j, k) = ((float)rand() / RAND_MAX) / scaleDown;
            }
        }
    }
}

void print(struct FFNN* ffnn) {
    puts("====================Feed Forward Neural Network=================");
    for (int l = 0; l < ffnn->numLayers; ++l) {
        printf("Layer %d\n", l);
        int numNodes = ffnn->layerSizes[l];
        if (l == 0) {
            printf("\t%d Inputs\n", numNodes);
            break;
        }
        int weightsPerNode = ffnn->layerSizes[l - 1];
        for (int j = 0; j < numNodes; ++j) {
            printf("\tNode %d\n", j);
            float bias = B(l, j);
            printf("\t\tB: %.2f", bias);
            for (int k = 0; k < weightsPerNode; ++k) {
                float weight = W(l, j, k);
                printf(", W%d: %.2f", k, weight);
            }
            putchar('\n');
        }
    }
    puts("===============================================================");
}

float quadraticCost(float* prediction, float* actual, int size) {
    float total = 0;
    for (int i = 0; i < size; ++i) {
        float diff = prediction[i] - actual[i];
        total += diff * diff;
    }
    return total;
}

// Call before forwardPass
// Inputs is shallow copied but not modified.
void setInput(struct FFNN* ffnn, float* inputs) {
    ffnn->forwardVals[0] = inputs;
}

// Requires a prior run of forwardPass
float* getOutput(struct FFNN* ffnn) {
    return ffnn->forwardVals[ffnn->numLayers - 1];
}

void forwardPass(struct FFNN* ffnn) {
    for (int l = 1; l < ffnn->numLayers; ++l) {
        for (int j = 0; j < ffnn->layerSizes[l]; ++j) {
            float accum = B(l, j);
            for (int k = 0; k < ffnn->layerSizes[l - 1]; ++k) {
                float input = A(l - 1, k);
                float weight = W(l, j, k);
                accum += input * weight;
            }
            accum = 1 / (1 + powf(M_E, -accum)); // sigmoid activation
            A(l, j) = accum;
        }
    }
}

// Requires a prior run of forwardPass
// Throughout this function, AToB represents the partial derivative of A with respect to B
struct NodeGradient** backwardPass(struct FFNN* ffnn, float* actual) {
    struct NodeGradient** gradient = malloc(sizeof(*gradient) * ffnn->numLayers);
    gradient -= 1; // same indexing as Nodes
    for (int l = 1; l < ffnn->numLayers; ++l) {
        gradient[l] = malloc(sizeof(**gradient) * ffnn->layerSizes[l]);
        for (int j = 0; j < ffnn->layerSizes[l]; ++j) {
            gradient[l][j].dWeights = malloc(sizeof((**gradient).dWeights[0])
                                                 * ffnn->layerSizes[l - 1]);
        }
    }

    {
        // calculate errors for output layer
        int l = ffnn->numLayers - 1;
        float* prediction = getOutput(ffnn);

        for (int j = 0; j < ffnn->layerSizes[l]; ++j) {
            float cToP = prediction[j] - actual[j];
            float pToZ = prediction[j] * (1 - prediction[j]); // derivative of sigmoid
            float cToZ = cToP * pToZ;
            gradient[l][j].dBias = cToZ;
        }
    }

    // calculate errors for hidden layers
    for (int l = ffnn->numLayers - 2; l > 0; --l) {
        for (int k = 0; k < ffnn->layerSizes[l]; ++k) {
            for (int j = 0; j < ffnn->layerSizes[l + 1]; ++j) {
                float pCToZ = gradient[l + 1][j].dBias;
                float zToA = W(l, j, k);
                float in = ffnn->forwardVals[l][k];
                float aToZ = in * (1 - in);
                float nCToZ = pCToZ * zToA * aToZ;
                gradient[l][k].dBias += nCToZ;
            }
        }
    }

    // apply errors to find sensitivity for weights and biases
    for (int l = 1; l < ffnn->numLayers; ++l) {
        for (int j = 0; j < ffnn->layerSizes[l]; ++j) {
            float sensitivity = gradient[l][j].dBias;
            for (int k = 0; k < ffnn->layerSizes[l - 1]; ++k) {
                float input = A(l - 1, k);
                gradient[l][j].dWeights[k] = sensitivity * input;
            }
        }
    }

    return gradient;
}

void applyGradient(struct FFNN* ffnn, struct NodeGradient** gradient, float learningRate) {
    for (int l = 1; l < ffnn->numLayers; ++l) {
        for (int j = 0; j < ffnn->layerSizes[l]; ++j) {
            B(l, j) -= learningRate * gradient[l][j].dBias;
            for (int k = 0; k < ffnn->layerSizes[l - 1]; ++k) {
                W(l, j, k) -= learningRate * gradient[l][j].dWeights[k];
            }
        }
    }
}

void stochasticTrain(struct FFNN* ffnn,
                    float** inputs, 
                    float** outputs, 
                    int trainingSetSize, 
                    float learningRate) {
    
    putchar('\n');
    for (int i = 0; i < trainingSetSize; ++i) {
        setInput(ffnn, inputs[i]);
        forwardPass(ffnn);
        struct NodeGradient** gradient = backwardPass(ffnn, outputs[i]);
        applyGradient(ffnn, gradient, learningRate);
        free(gradient + 1);

        float* guess = getOutput(ffnn);
        float cost = quadraticCost(guess, outputs[i], ffnn->layerSizes[ffnn->numLayers - 1]);

        printf("\033[A\33[2K\rCost:  %3.3f\n", cost);
    }
}