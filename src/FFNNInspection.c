#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "FFNNInspection.h"
#include "FFNN.h"
#include "asciiPixel.h"

void inspect(struct FFNN* ffnn) {
    puts("====================Feed Forward Neural Network=================");
    for (int l = 0; l < ffnn->numLayers; ++l) {
        printf("Layer %d\n", l);
        int numNodes = ffnn->layerSizes[l];
        if (l == 0) {
            printf("\t%d Inputs\n", numNodes);
            continue;
        } else if (ffnn->classifier && l == ffnn->numLayers - 1) {
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

/**
 * Returns index of greatest value
 */
int maxIndex(float* in, int num) {
    float max = -FLT_MAX;
    int index = -1;
    for (int i = 0; i < num; ++i) {
        if (in[i] > max) {
            max = in[i];
            index = i;
        }
    }
    return index;
}

void test(struct FFNN* ffnn, float** inputs, float** outputs, int setSize) {
    putchar('\n');
    int errorCount = 0;
    int numOutputs = ffnn->layerSizes[ffnn->numLayers - 1];
    for (int i = 0; i < setSize; ++i) {
        setInput(ffnn, inputs[i]);
        forwardPass(ffnn);
        int guessIndex = maxIndex(getOutput(ffnn), numOutputs);
        int goodIndex = maxIndex(outputs[i], numOutputs);
        if (guessIndex != goodIndex) {
            errorCount += 1;
        }
        printf("\033[A\33[2K\rError Rate: %.2f%% (%d)\n",
                            100 * (float)errorCount / (i + 1), i + 1);
    }
}


/**
 * Simplified for getting receptive field for single output node, rather than entire network
 */
void populateOutputReceptiveField(float* receptiveField,
                                    int outputNode,
                                    struct FFNN* ffnn) {
    int numInputs = ffnn->layerSizes[0];
    float* inputs = calloc(numInputs, sizeof(*inputs));
    for (int i = 0; i < numInputs; ++i) {
        inputs[i] = 1;
        setInput(ffnn, inputs);
        for (int l = 1; l < ffnn->numLayers; ++l) {
            for (int j = 0; j < ffnn->layerSizes[l]; ++j) {
                if (l == ffnn->numLayers - 1 && j != outputNode) {
                    continue;
                }
                float accum = 0;
                for (int k = 0; k < ffnn->layerSizes[l - 1]; ++k) {
                    float input = A(l - 1, k);
                    float weight = W(l, j, k);
                    accum += input * weight;
                }
                A(l, j) = accum;
                if (l == ffnn->numLayers - 1) {
                    receptiveField[i] = accum;
                }
            }
        }
        inputs[i] = 0;
    }
    free(inputs);
}

float*** allocReceptiveFields(struct FFNN* ffnn) {
    int numInputs = ffnn->layerSizes[0];
    float*** fields = malloc(sizeof(*fields) * ffnn->numLayers);
    for (int l = 0; l < ffnn->numLayers; ++l) {
        int numNodes = ffnn->layerSizes[l];
        fields[l] = malloc(sizeof(**fields) * numNodes);
        for (int j = 0; j < numNodes; ++j) {
            if (l!=0) {
                fields[l][j] = malloc(sizeof(***fields) * numInputs);
            } else {
                // inputs nodes only set 1 pixel
                fields[l][j] = calloc(numInputs, sizeof(***fields));
            }
        }
    }
    return fields;
}

void populateReceptiveFields(float*** fields, struct FFNN* ffnn) {
    int numInputs = ffnn->layerSizes[0];
    float* inputs = calloc(numInputs, sizeof(*inputs));
    for (int i = 0; i < numInputs; ++i) {
        inputs[i] = 1;
        fields[0][i][i] = 1; // input nodes have a receptive field of 1 pixel
        setInput(ffnn, inputs);
        
        // modified forwardPass
        // - removes biases
        // - removes activation
        // - logs the weighted inputs
        for (int l = 1; l < ffnn->numLayers; ++l) {
            for (int j = 0; j < ffnn->layerSizes[l]; ++j) {
                float accum = 0;
                for (int k = 0; k < ffnn->layerSizes[l - 1]; ++k) {
                    float input = A(l - 1, k);
                    float weight = W(l, j, k);
                    accum += input * weight;
                }
                fields[l][j][i] = accum;
                A(l, j) = accum;
            }
        }

        inputs[i] = 0;
    }

    free(inputs);
}

void freeReceptiveFields(float*** fields, struct FFNN* ffnn) {
    for (int l = 0; l < ffnn->numLayers; ++l) {
        for (int j = 0; j < ffnn->layerSizes[l]; ++j) {
            free(fields[l][j]);
        }
        free(fields[l]);
    }
    free(fields);
}
