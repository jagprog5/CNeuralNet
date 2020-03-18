#include <stdarg.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "FFNN.h"

struct FFNN* alloc(int numLayers, int* layerSizes) {
    assert(numLayers > 1);
    for (int i = 0; i < numLayers; ++i) {
        assert(layerSizes[i] > 0);
    }

    struct FFNN *ffnn = malloc(sizeof(struct FFNN));
    ffnn->numLayers = numLayers;

    ffnn->layerSizes = malloc(sizeof(int) * numLayers);
    for (int i = 0; i < numLayers; ++i) {
        // deep copy for safety
        ffnn->layerSizes[i] = layerSizes[i];
    }

    // first layer only contains inputs, not Nodes
    ffnn->nodes = malloc(sizeof(struct Node*) * (numLayers - 1));
    for (int layer = 1; layer < numLayers; ++layer) {
        int numNodes = ffnn->layerSizes[layer];
        int weightsPerNode = ffnn->layerSizes[layer - 1];

        ffnn->nodes[nodesIndexFFNN] = malloc(sizeof(struct Node) * numNodes);
        for (int i = 0; i < numNodes; ++i) {
            ffnn->nodes[nodesIndexFFNN][i].weights = malloc(sizeof(float) * weightsPerNode);
        }
    }

    ffnn->forwardVals = malloc(sizeof(float*) * numLayers);
    for (int layer = 0; layer < numLayers; ++layer) {
        ffnn->forwardVals[layer] = malloc(sizeof(float) * ffnn->layerSizes[layer]);
    }

    // Only required for training
    ffnn->forwardLog = malloc(sizeof(struct ForwardLog*) * (numLayers - 1));
    for (int layer = 1; layer < numLayers; ++layer) {
        int numNodes = ffnn->layerSizes[layer];
        int inputsPerNode = ffnn->layerSizes[layer - 1];

        ffnn->forwardLog[nodesIndexFFNN] = malloc(sizeof(struct ForwardLog) * numNodes);
        for (int i = 0; i < numNodes; ++i) {
            ffnn->forwardLog[nodesIndexFFNN][i].nodeInputs = malloc(sizeof(float) * inputsPerNode);
        }
    }

    return ffnn;
}

void randomize(struct FFNN* ffnn) {
    assert(ffnn);
    for (int layer = 1; layer < ffnn->numLayers; ++layer) {
        int numNodes = ffnn->layerSizes[layer];
        int weightsPerNode = ffnn->layerSizes[layer - 1];
        for (int i = 0; i < numNodes; ++i) {
            ffnn->nodes[nodesIndexFFNN][i].bias = (float)rand() / RAND_MAX;
            for (int j = 0; j < weightsPerNode; ++j) {
                ffnn->nodes[nodesIndexFFNN][i].weights[j] = (float)rand() / RAND_MAX;
            }
        }
    }
}

void print(struct FFNN* ffnn) {
    assert(ffnn);
    printf("Feed Forward Neural Network\n");
    for (int layer = 0; layer < ffnn->numLayers; ++layer) {
        printf("Layer %d\n", layer);
        int numNodes = ffnn->layerSizes[layer];
        int weightsPerNode = ffnn->layerSizes[layer - 1];
        for (int i = 0; i < numNodes; ++i) {
            if (layer==0) {
                printf("\t%d Inputs\n", numNodes);
                break;
            }
            printf("\tNode %d\n", i);
            float bias = ffnn->nodes[nodesIndexFFNN][i].bias;
            printf("\t\tB: %.2f", bias);
            for (int j = 0; j < weightsPerNode; ++j) {
                float weight = ffnn->nodes[nodesIndexFFNN][i].weights[j];
                printf(", W%d: %.2f", j, weight);
            }
            putchar('\n');
        }
    }
}

void setInput(struct FFNN* ffnn, float* inputs) {
    assert(ffnn);
    free(ffnn->forwardVals[0]);
    ffnn->forwardVals[0] = inputs;
}

float sigmoid(float in) {
    return 1 / (1 + powf(M_E, -in));
}

void forwardPass(struct FFNN* ffnn) {
    for (int layer = 1; layer < ffnn->numLayers; ++layer) {
        int numNodes = ffnn->layerSizes[layer];
        int weightsPerNode = ffnn->layerSizes[layer - 1];
        for (int i = 0; i < numNodes; ++i) {
            float accum = 0;
            accum += ffnn->nodes[nodesIndexFFNN][i].bias;
            for (int j = 0; j < weightsPerNode; ++j) {
                accum += ffnn->nodes[nodesIndexFFNN][i].weights[j]
                            * ffnn->forwardVals[layer - 1][j];
            }
            accum = sigmoid(accum);
            ffnn->forwardVals[layer][i] = accum;
        }
    }
}

float* getOutput(struct FFNN* ffnn) {
    assert(ffnn);
    return ffnn->forwardVals[ffnn->numLayers - 1];
}

float quadraticCost(float* prediction, float* actual, int size) {
    float total = 0;
    for (int i = 0; i < size; ++i) {
        float diff = prediction[i] - actual[i];
        total += diff * diff;
    }
    return total;
}

struct NodeGradient** backwardPass(struct FFNN* ffnn, float* actual) {
    float* output = getOutput(ffnn);

    struct NodeGradient** gradient = malloc(sizeof(struct NodeGradient*) * (ffnn->numLayers - 1));
    // iterate from output to 1st hidden layer
    for (int layer = ffnn->numLayers; layer > 0; --layer) {
        gradient[nodesIndexFFNN] = malloc(sizeof(struct NodeGradient) * (ffnn->layerSizes[layer]));
        
        int numNodes = ffnn->layerSizes[layer];
        int weightsPerNode = ffnn->layerSizes[layer - 1];
        for (int i = 0; i < numNodes; ++i) {
            // partial derivative of quadratic cost to prediction
            float cToP = 2 * (output[i] - actual[i]);

            // derivative of sigmoid is sig(x) * (1 - sig(x))
            float sigOut = ffnn->forwardVals[layer][i];
            // partial derivative of prediction to weighted sum
            float pToS = sigOut * (1 - sigOut);
                //dSigmoid(ffnn->forwardLog[nodesIndexFFNN][i].weightedSum);

            float cToS = cToP * pToS; // chain rule
            for (int j = 0; j < weightsPerNode; ++j) {
                // partial derivative of weight sum to weight
                float sToW = ffnn->forwardLog[nodesIndexFFNN][i].nodeInputs[j];
                float cToW = cToS * sToW;
                gradient[nodesIndexFFNN][i].dWeights[j] = cToW;
            }
            float sToB = 1; // keeping for completeness
            float cToB = cToS * sToB;
            gradient[nodesIndexFFNN][i].dBias = cToB;

            for (int fLayer = layer; fLayer < ffnn->numLayers; ++fLayer) {
                
            }
        }
    }
    return gradient;
}

