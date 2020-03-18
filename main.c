#include <stdio.h>
#include <stdlib.h>
#include "FFNN.h"
int main() {
    // TODO create make file
    // gcc main.c FFNN.h FFNN.c -o ./a.exe -lm && ./a.exe

    int layerSizes[5] = {3, 4, 4, 4, 3};
    struct FFNN* ffnn = alloc(4, layerSizes);
    randomize(ffnn);

    int sample = 3;
    float inputs[3][4] =       {{1, 0, 0, 0},
                                {1, 1, 0, 0},
                                {1, 1, 1, 0}};
    float idealOutputs[3][4] = {{0, 1, 1, 1},
                                {0, 0, 1, 1},
                                {0, 0, 0, 1}};
    for (int i = 0; i < 100000; ++i) {
        int index = rand() % 3;
        setInput(ffnn, inputs[index]);
        forwardPass(ffnn);
        struct NodeGradient** gradient = backwardPass(ffnn, idealOutputs[index]);
        applyGradient(ffnn, gradient, 1);
        if (i % 4269 == 0) {
            float* output = getOutput(ffnn);
            printf("Input:%d, Out:{%f,%f,%f,%f}\n", index, output[0], 
                                    output[1], output[2], output[3]);
        }
    }

    return 0;
}