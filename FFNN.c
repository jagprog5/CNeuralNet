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

void randomize(struct FFNN* ffnn) {
    for (int l = 1; l < ffnn->numLayers; ++l) {
        float scaleDown = ffnn->layerSizes[l - 1] + 1;
        for (int j = 0; j < ffnn->layerSizes[l]; ++j) {
            B(l, j) = ((float)rand() / RAND_MAX) / scaleDown;
            for (int k = 0; k < ffnn->layerSizes[l - 1]; ++k) {
                W(l, j, k) = ((float)rand() / RAND_MAX) / scaleDown;
            }
        }
    }
}

// vals is deep copied
void setNetwork(struct FFNN* ffnn, float** vals) {
    vals -= 1;
    for (int l = 1; l < ffnn->numLayers; ++l) {
        int neuronOffset = ffnn->layerSizes[l - 1] + 1;
        for (int j = 0; j < ffnn->layerSizes[l]; ++j) {
            B(l, j) = vals[l][j * neuronOffset];
            for (int k = 0; k < ffnn->layerSizes[l - 1]; ++k) {
                W(l, j, k) = vals[l][k + 1 + j * neuronOffset];
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
            continue;
        }
        int weightsPerNode = ffnn->layerSizes[l - 1];
        for (int j = 0; j < numNodes; ++j) {
            printf("\tNode %d\n", j);
            float bias = B(l, j);
            printf("\t\tB: %f", bias);
            for (int k = 0; k < weightsPerNode; ++k) {
                float weight = W(l, j, k);
                printf(", W%d: %f", k, weight);
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
// Returns a shallow copy that will be modified after a call of forward pass
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

    for (int l = ffnn->numLayers - 1; l > 0; --l) {
        for (int k = 0; k < ffnn->layerSizes[l]; ++k) {
            gradient[l][k].dBias = 0;
            float neuronOutput = ffnn->forwardVals[l][k];
            if (l == ffnn->numLayers - 1) {
                gradient[l][k].dBias = actual[k] - neuronOutput; // output neurons
            } else {
                for (int j = 0; j < ffnn->layerSizes[l + 1]; ++j) {
                    float error = gradient[l + 1][j].dBias;
                    float weight = W(l + 1, j, k);
                    gradient[l][k].dBias += error * weight;
                }
            }
            gradient[l][k].dBias *= neuronOutput * (1 - neuronOutput); // derivative of sigmoid
        }
    }

    // apply errors to find sensitivity for weights
    for (int l = 1; l < ffnn->numLayers; ++l) {
        for (int j = 0; j < ffnn->layerSizes[l]; ++j) {
            float error = gradient[l][j].dBias;
            for (int k = 0; k < ffnn->layerSizes[l - 1]; ++k) {
                float input = A(l - 1, k);
                gradient[l][j].dWeights[k] = error * input;
            }
        }
    }

    return gradient;
}

void applyGradient(struct FFNN* ffnn, struct NodeGradient** gradient, float learningRate) {
    for (int l = 1; l < ffnn->numLayers; ++l) {
        for (int j = 0; j < ffnn->layerSizes[l]; ++j) {
            B(l, j) += learningRate * gradient[l][j].dBias;
            for (int k = 0; k < ffnn->layerSizes[l - 1]; ++k) {
                W(l, j, k) += learningRate * gradient[l][j].dWeights[k];
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