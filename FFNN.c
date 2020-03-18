#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "FFNN.h"

struct FFNN* alloc(int numLayers, int* layerSizes) {
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
            ffnn->forwardLog[nodesIndexFFNN][i].cToI = malloc(sizeof(float) * inputsPerNode);
            ffnn->forwardLog[nodesIndexFFNN][i].nodeInputs = malloc(sizeof(float) * inputsPerNode);
        }
    }

    return ffnn;
}

void randomize(struct FFNN* ffnn) {
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
    puts("====================Feed Forward Neural Network=================");
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
    puts("===============================================================");
}

void setInput(struct FFNN* ffnn, float* inputs) {
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
                float input = ffnn->forwardVals[layer - 1][j];
                accum += ffnn->nodes[nodesIndexFFNN][i].weights[j] * input;
                ffnn->forwardLog[nodesIndexFFNN][i].nodeInputs[j] = input;
            }
            accum = sigmoid(accum);
            ffnn->forwardVals[layer][i] = accum;
        }
    }
}

// Requires a prior run of forwardPass
float* getOutput(struct FFNN* ffnn) {
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

// Requires a prior run of forwardPass
struct NodeGradient** backwardPass(struct FFNN* ffnn, float* actual) {
    // Throughout this function, AToB represents the partial derivative of A with respect to B

    struct NodeGradient** gradient = malloc(sizeof(struct NodeGradient*) * (ffnn->numLayers - 1));

    // cost to predictions (predictions are the outputs from the output layer)
    float* prediction = getOutput(ffnn);
    int outputLength = ffnn->layerSizes[ffnn->numLayers - 1];
    float cToP[outputLength];
    for (int i = 0; i < outputLength; ++i) {
        cToP[i] = prediction[i] - actual[i];
        //cToP[i] *= 2; // leaving for completeness
    }

    // iterate from output to 1st hidden layer
    for (int layer = ffnn->numLayers - 1; layer > 0; --layer) {
        int numNodes = ffnn->layerSizes[layer];
        gradient[nodesIndexFFNN] = malloc(sizeof(struct NodeGradient) * numNodes);
        int weightsPerNode = ffnn->layerSizes[layer - 1];
        for (int i = 0; i < numNodes; ++i) {
            gradient[nodesIndexFFNN][i].dWeights = malloc(sizeof(float) * weightsPerNode);
            // derivative of sigmoid is sig(x) * (1 - sig(x))
            float sigOut = ffnn->forwardVals[layer][i];
            // partial derivative of output to weighted sum
            float oToS = sigOut * (1 - sigOut);

            for (int j = 0; j < weightsPerNode; ++j) {
                    // corresponding weight for input
                float sToI = ffnn->nodes[nodesIndexFFNN][i].weights[j];
                float oToI = oToS * sToI;

                float priorCToI;
                if (layer == ffnn->numLayers - 1) {
                    // currently in output layer
                    priorCToI = cToP[i];
                } else {
                    priorCToI = 0;
                    for (int k = 0; k < ffnn->layerSizes[layer + 1]; ++k) {
                        priorCToI += ffnn->forwardLog[nodesIndexFFNN + 1][k].cToI[i];
                    }
                }
                float cToI = oToI * priorCToI;
                ffnn->forwardLog[nodesIndexFFNN][i].cToI[j] = cToI;

                float sToW = ffnn->forwardLog[nodesIndexFFNN][i].nodeInputs[j];
                float oToW = oToS * sToW; // output to weight
                float cToW = oToW * priorCToI; // cost function to weight
                gradient[nodesIndexFFNN][i].dWeights[j] =  cToW;

                float sToB = 1; // keeping for completeness
                float cToB = priorCToI * oToS * sToB;
                gradient[nodesIndexFFNN][i].dBias = cToB;
            }
        }
    }
    return gradient;
}

void applyGradient(struct FFNN* ffnn, struct NodeGradient** gradient, float learningRate) {
    for (int layer = 1; layer < ffnn->numLayers; ++layer) {
        for (int i = 0; i < ffnn->layerSizes[layer]; ++i) {
            ffnn->nodes[nodesIndexFFNN][i].bias -= learningRate * gradient[nodesIndexFFNN][i].dBias;
            for (int k = 0; k < ffnn->layerSizes[layer - 1]; ++k) {
                ffnn->nodes[nodesIndexFFNN][i].weights[k] -= learningRate * gradient[nodesIndexFFNN][i].dWeights[k];
            }
        }
    }
}