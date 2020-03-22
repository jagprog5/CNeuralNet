#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "FFNN.h"

/**
 * layerSizes is shallow copied, but not modified
 */
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

    setRegressional(ffnn);

    return ffnn;
}

/**
 * Indicates usage of softmax on output nodes, and cross entropy loss
 */
void setCategorical(struct FFNN* ffnn) {
    ffnn->categorical = 1;
}

/**
 * Default mode.
 * Uses sigmoid activation throughout, and MSE cost.
 */
void setRegressional(struct FFNN* ffnn) {
    ffnn->categorical = 0;
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
        } else if (ffnn->categorical && l == ffnn->numLayers - 1) {
            puts("\tUsing Softmax Activation");
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
    // could multiply total by 2 for completeness
    return total;
}

float crossEntropyCost(float* prediction, float* actual, int size) {
    float total = 0;
    for (int i = 0; i < size; ++i) {
        // could be base 2
        total += actual[i] * logf(prediction[i]);
    }
    total *= -1;
}

/**
 * inputs is shallow copied but not modified
 * setInput should be called prior to forwardPass
 */
void setInput(struct FFNN* ffnn, float* inputs) {
    ffnn->forwardVals[0] = inputs;
}
 
/**
 * Returns a shallow copy whose content will be modified after every call of forward pass
 * Requires a prior run of forwardPass
 */
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
            // don't apply sigmoid if softmax enabled and in output layer
            if (!ffnn->categorical || l != ffnn->numLayers - 1) {
                accum = 1 / (1 + powf(M_E, -accum)); // sigmoid activation
            }
            A(l, j) = accum;
        }
    }
    if (ffnn->categorical) {
        int l = ffnn->numLayers - 1;
        int n = ffnn->layerSizes[l];
        float denominator = 0;
        for (int j = 0; j < n; ++j) {
            A(l, j) = powf(M_E, -A(l, j));
            denominator += A(l, j);
        }
        for (int j = 0; j < n; ++j) {
            A(l, j) /= denominator;
        }
    }
}

/**
 * Requires a prior run of forwardPass.
 * Note that the negative gradient is returned (the direction that will minimize the cost).
 */
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
                if (ffnn->categorical) {
                    gradient[l][k].dBias = -actual[k] / neuronOutput
                                                +    (1 - actual[k]) / (1 - neuronOutput);
                } else {
                    gradient[l][k].dBias = actual[k] - neuronOutput; // output neurons
                }
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

/**
 * Stochastic gradient descent
 */
void SGD(struct FFNN* ffnn,
                    float** inputs, 
                    float** outputs, 
                    int trainingSetSize, 
                    float learningRate,
                    int epochs) {
    putchar('\n');
    for (int e = 0; e < epochs; ++e) {
        for (int i = 0; i < trainingSetSize; ++i) {
            setInput(ffnn, inputs[i]);
            forwardPass(ffnn);
            struct NodeGradient** gradient = backwardPass(ffnn, outputs[i]);
            applyGradient(ffnn, gradient, learningRate);
            free(gradient + 1);

            float* guess = getOutput(ffnn);
            float cost;
            if (ffnn->categorical) {
                cost = crossEntropyCost(guess, outputs[i], ffnn->layerSizes[ffnn->numLayers - 1]);
            } else {
                cost = quadraticCost(guess, outputs[i], ffnn->layerSizes[ffnn->numLayers - 1]);
            }

            printf("\033[A\33[2K\rCost: %3.3f\n", e, cost);

        }
    }
}