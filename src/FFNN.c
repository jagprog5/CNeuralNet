#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "FFNN.h"
#include "printReducer.h"

/**
 * layerSizes is shallow copied, but not modified. It will also not be freed by freeFFNN
 */
struct FFNN* allocFFNN(int numLayers, int* layerSizes) {
    struct FFNN *ffnn = malloc(sizeof(*ffnn));
    ffnn->numLayers = numLayers;

    ffnn->layerSizes = malloc(sizeof(*ffnn->layerSizes) * numLayers);
    ffnn->layerSizes = layerSizes;
    ffnn->nodes = allocNodes(numLayers, layerSizes);

    ffnn->forwardVals = malloc(sizeof(*ffnn->forwardVals) * numLayers);
    for (int l = 1; l < numLayers; ++l) {
        // for layer 0, inputs are given by setInputs
        ffnn->forwardVals[l] = malloc(sizeof(**ffnn->forwardVals) * ffnn->layerSizes[l]);
    }

    setRegressional(ffnn);

    return ffnn;
}

/**
 * Indicates usage of softmax on output nodes, and cross entropy loss
 */
void setClassifier(struct FFNN* ffnn) {
    ffnn->classifier = 1;
}

/**
 * Default mode.
 * Uses sigmoid activation throughout (including output nodes), and MSE loss.
 */
void setRegressional(struct FFNN* ffnn) {
    ffnn->classifier = 0;
}

void randomize(struct FFNN* ffnn) {
    static int needSeeding = 1;
    if (needSeeding) {
        needSeeding = 0;
        srand(time(NULL));
    }
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

/**
 * vals is deep copied.
 * vals[l] points to the weights and biases for layer l + 1 (since l=0 has no weights or biases)
 */
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

float quadraticLoss(float* prediction, float* actual, int size) {
    float total = 0;
    for (int i = 0; i < size; ++i) {
        float diff = prediction[i] - actual[i];
        total += diff * diff;
    }
    return total;
}

float crossEntropyLoss(float* prediction, float* actual, int size) {
    float total = 0;
    for (int i = 0; i < size; ++i) {
        total += actual[i] * logf(prediction[i]);
    }
    total *= -1;
}

/**
 * inputs is shallow copied but not modified
 * setInput should be called prior to forwardPass
 * Does not free previous inputs
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
            if (!ffnn->classifier || l != ffnn->numLayers - 1) {
                accum = 1 / (1 + powf(M_E, -accum)); // sigmoid activation
            }
            A(l, j) = accum;
        }
    }
    if (ffnn->classifier) {
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
 * Note that the negative gradient is returned (the direction that will minimize the loss).
 */
struct Node** backwardPass(struct FFNN* ffnn, float* actual) {
    struct Node** gradient = allocNodes(ffnn->numLayers, ffnn->layerSizes);

    for (int l = ffnn->numLayers - 1; l > 0; --l) {
        for (int k = 0; k < ffnn->layerSizes[l]; ++k) {
            gradient[l][k].bias = 0;
            float neuronOutput = ffnn->forwardVals[l][k];
            if (l == ffnn->numLayers - 1) {
                if (ffnn->classifier) {
                    gradient[l][k].bias = -actual[k] / neuronOutput
                                                +    (1 - actual[k]) / (1 - neuronOutput);
                } else {
                    gradient[l][k].bias = actual[k] - neuronOutput; // output neurons
                }
            } else {
                for (int j = 0; j < ffnn->layerSizes[l + 1]; ++j) {
                    float error = gradient[l + 1][j].bias;
                    float weight = W(l + 1, j, k);
                    gradient[l][k].bias += error * weight;
                }
            }
            gradient[l][k].bias *= neuronOutput * (1 - neuronOutput); // derivative of sigmoid
        }
    }

    // apply errors to find sensitivity for weights
    for (int l = 1; l < ffnn->numLayers; ++l) {
        for (int j = 0; j < ffnn->layerSizes[l]; ++j) {
            float error = gradient[l][j].bias;
            for (int k = 0; k < ffnn->layerSizes[l - 1]; ++k) {
                float input = A(l - 1, k);
                gradient[l][j].weights[k] = error * input;
            }
        }
    }

    return gradient;
}

void applyGradient(struct FFNN* ffnn, struct Node** gradient, float learningRate) {
    for (int l = 1; l < ffnn->numLayers; ++l) {
        for (int j = 0; j < ffnn->layerSizes[l]; ++j) {
            B(l, j) += learningRate * gradient[l][j].bias;
            for (int k = 0; k < ffnn->layerSizes[l - 1]; ++k) {
                W(l, j, k) += learningRate * gradient[l][j].weights[k];
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
                    float learningRate) {
    putchar('\n');
    for (int i = 0; i < trainingSetSize; ++i) {
        setInput(ffnn, inputs[i]);
        forwardPass(ffnn);
        struct Node** gradient = backwardPass(ffnn, outputs[i]);
        applyGradient(ffnn, gradient, learningRate);
        freeNodes(gradient, ffnn->numLayers, ffnn->layerSizes);

        float* guess = getOutput(ffnn);
        float loss;
        if (ffnn->classifier) {
            loss = crossEntropyLoss(guess, outputs[i], ffnn->layerSizes[ffnn->numLayers - 1]);
        } else {
            loss = quadraticLoss(guess, outputs[i], ffnn->layerSizes[ffnn->numLayers - 1]);
        }
        prt_redu(i, 300, printf("\033[A\33[2K\rTraining: %d\n", i);)
    }
}

struct Node** allocNodes(int numLayers, int* layerSizes) {
    struct Node** nodes = malloc(sizeof(*nodes) * numLayers);
    nodes -= 1;
    for (int l = 1; l < numLayers; ++l) {
        nodes[l] = malloc(sizeof(**nodes) * layerSizes[l]);
        for (int j = 0; j < layerSizes[l]; ++j) {
            nodes[l][j].weights = malloc(sizeof((**nodes).weights[0]) * layerSizes[l - 1]);
        }
    }
    return nodes;
}

void freeNodes(struct Node** nodes, int numLayers, int* layerSizes) {
    for (int l = 1; l < numLayers; ++l) {
        for (int j = 0; j < layerSizes[l]; ++j) {
            free(nodes[l][j].weights);
        }
        free(nodes[l]);
    }
    nodes += 1;
    free(nodes);
}

/**
 * This does not free layerSizes, or anything passed into setInput
 */
void freeFFNN(struct FFNN* ffnn) {
    freeNodes(ffnn->nodes, ffnn->numLayers, ffnn->layerSizes);
    for (int l = 1; l < ffnn->numLayers; ++l) {
        // Don't free forward vals at inputs!
        free(ffnn->forwardVals[l]);
    }
    free(ffnn->forwardVals);
}
