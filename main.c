#include <stdio.h>
#include <stdlib.h>
#include "FFNN.h"
int main() {
    // TODO create make file
    // gcc main.c FFNN.h FFNN.c -o ./a.exe -lm && ./a.exe

    int layerSizes[4] = {3, 4, 4, 3};
    struct FFNN* ffnn = alloc(4, layerSizes);
    randomize(ffnn);
    print(ffnn);

    float inputs[4] = {1, 1, 1, 1};
    setInput(ffnn, inputs);
    forwardPass(ffnn);
    float* out = getOutput(ffnn);
    printf("%.2f\n", out[0]);
    return 0;
}