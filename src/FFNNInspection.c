#include <ncurses.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "FFNNInspection.h"
#include "FFNN.h"

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

/**
 * Simplified for getting receptive field for single output node, rather than entire network.
 * flag and flag2 skip computations thay aren't needed
 */
void populateOutputReceptiveField(float* receptiveField,
                                    int outputNode,
                                    struct FFNN* ffnn) {
    int numInputs = ffnn->layerSizes[0];
    float* inputs = calloc(numInputs, sizeof(*inputs));
    for (int i = 0; i < numInputs; ++i) {
        inputs[i] = 1;
        setInput(ffnn, inputs);
        // modified forwardPass
        // - removes biases
        // - removes activation
        // - logs the weighted inputs
        for (int l = 1; l < ffnn->numLayers; ++l) {
            int flag = l == ffnn->numLayers - 1;
            for (int j = 0; j < ffnn->layerSizes[l]; ++j) {
                if (flag) {
                    j = outputNode;
                }
                float accum = 0;
                int flag2 = l == 1;
                for (int k = 0; k < ffnn->layerSizes[l - 1]; ++k) {
                    if (flag2) {
                        k = i;
                    }
                    float input = A(l - 1, k);
                    float weight = W(l, j, k);
                    accum += input * weight;
                    if (flag2) {
                        break;
                    }
                }
                A(l, j) = accum;
                if (flag) {
                    receptiveField[i] = accum;
                    break;
                }
            }
        }
        inputs[i] = 0;
    }
    free(inputs);
}