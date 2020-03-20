#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include "FFNN.h"
#include "MNISTRead.h"

void MNISTVisualStochasticTrain(struct FFNN* ffnn,
                    float** inputs, 
                    float** outputs, 
                    int trainingSetSize, 
                    float learningRate,
                    int width,
                    int height);

int main() {
    // TODO: create make file
    // gcc main.c FFNN.h FFNN.c MNISTRead.h MNISTRead.c -o ./a.exe -lm && ./a.exe
    
    int layers[2] = {2, 2};
    struct FFNN* ffnn = alloc(3, layers);
    randomize(ffnn);

    // setInput(ffnn, imgs[0]);
    // forwardPass(ffnn);
    // backwardPass(ffnn, labels[0]);
    // stochasticTrain(ffnn, imgs, labels, numImages, 0.1f);

    return 0;
}

/*
int main() {
    int numImages;
    int width;
    int height;
    float** imgs = readMNISTImages(&numImages, &width, &height);
    float** labels = readMNISTLabels(&numImages);
    
    int inputLayerSize = width * height;
    int layers[5] = {inputLayerSize, 
                        inputLayerSize << 2,
                        inputLayerSize << 4,
                        inputLayerSize << 6,
                        10};
    struct FFNN* ffnn = alloc(3, layers);
    randomize(ffnn);
    MNISTVisualStochasticTrain(ffnn, imgs, labels, numImages, 0.01f, width, height);
    return 0;
}

void MNISTVisualStochasticTrain(struct FFNN* ffnn,
                    float** inputs, 
                    float** outputs, 
                    int trainingSetSize, 
                    float learningRate,
                    int width,
                    int height) {

    puts("========Training========");
    int numLines = height + 5;
    for (int i = 0; i < numLines; ++i) {
        putchar('\n');
    }
    
    for (int i = 0; i < trainingSetSize; ++i) {
        sleep(1);
        setInput(ffnn, inputs[i]);
        forwardPass(ffnn);
        struct NodeGradient** gradient = backwardPass(ffnn, outputs[i]);
        applyGradient(ffnn, gradient, learningRate);
        free(gradient + 1);

        // =========
        // Make sure to have the terminal window tall enough


        for (int j = 0; j < numLines; ++j) {
            printf("\033[A\33[2K\r");
        }

        float* guess = getOutput(ffnn);
        float cost = quadraticCost(guess, outputs[i], 10);

        char* imgStr = getImgStr(inputs[i], width, height);
        puts(imgStr);
        puts("       0123456789");
        printf("Label: ");
        for (int j = 0; j < 10; ++j) {
            putchar(shade(outputs[i][j]));
        }
        putchar('\n');
        printf("Guess: ");
        for (int j = 0; j < 10; ++j) {
            putchar(shade(guess[j]));
        }
        putchar('\n');
        printf("Cost: %.3f\n", cost);
    }
    putchar('\a');
}

*/