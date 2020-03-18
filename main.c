#include <stdio.h>
#include <stdlib.h>
#include "FFNN.h"
int main() {
    // TODO: create make file
    // gcc main.c FFNN.h FFNN.c -o ./a.exe -lm && ./a.exe

    int layerSizes[6] = {4, 4, 5, 6, 6, 4};
    struct FFNN* ffnn = alloc(4, layerSizes);
    randomize(ffnn);

    int sample = 6;
    float inputs[6][4] =       {{0, 0, 0, 0},
                                {0, 0, 0, 1},
                                {0, 0, 1, 0},
                                {0, 0, 1, 1},
                                {0, 1, 0, 0},
                                {1, 0, 0, 0}};
    float idealOutputs[6][4] = {{1, 1, 1, 1},
                                {1, 1, 1, 1},
                                {0, 1, 1, 1},
                                {1, 0, 1, 1},
                                {1, 1, 0, 1},
                                {0, 0, 0, 0}};
    for (int i = 0; i < 100000; ++i) {
        int index = rand() % sample;
        setInput(ffnn, inputs[index]);
        forwardPass(ffnn);
        struct NodeGradient** gradient = backwardPass(ffnn, idealOutputs[index]);
        applyGradient(ffnn, gradient, 0.1f);
        if (i % 4269 == 0) {
            float* output = getOutput(ffnn);
            float cost = quadraticCost(output, idealOutputs[index], 4);
            printf("%f\n", cost);
            // printf("Input:%d, Out:{%f,%f,%f,%f}\n", index, output[0], 
            //                         output[1], output[2], output[3]);
        }
    }

    return 0;
}