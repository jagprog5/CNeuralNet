#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include "FFNN.h"
#include "MNISTRead.h"

void MNISTVisualSGD(struct FFNN* ffnn,
                    float** inputs, 
                    float** outputs, 
                    int trainingSetSize, 
                    float learningRate,
                    int width,
                    int height);

int main() {
    int numImages;
    int width;
    int height;
    float** imgs = readMNISTImages(&numImages, &width, &height);
    float** labels = readMNISTLabels(&numImages);
    
    int inputLayerSize = width * height;
    int layerSizes[3] = {inputLayerSize, 
                        32,
                        10};
    struct FFNN* ffnn = allocFFNN(3, layerSizes);
    randomize(ffnn);
    setCategorical(ffnn);
    MNISTVisualSGD(ffnn, imgs, labels, numImages, 0.01f, width, height);
    // SGD(ffnn, imgs, labels, numImages, 0.01f);
    freeFFNN(ffnn);
    for (int i = 0; i < numImages; ++i) {
        free(imgs[i]);
        free(labels[i]);
    }
    free(imgs);
    free(labels);

    return 0;
}


void MNISTVisualSGD(struct FFNN* ffnn,
                    float** inputs, 
                    float** outputs, 
                    int trainingSetSize, 
                    float learningRate,
                    int width,
                    int height) {

    puts("========Training========");
    int numLines = height + 6;
    for (int i = 0; i < numLines; ++i) {
        putchar('\n');
    }
    
    for (int i = 0; i < trainingSetSize; ++i) {
        setInput(ffnn, inputs[i]);
        forwardPass(ffnn);
        struct Node** gradient = backwardPass(ffnn, outputs[i]);
        applyGradient(ffnn, gradient, learningRate);

        // =========
        // Everything after this is a visual addition to the SGD function in FFNN.c
        // Make sure to have the terminal window tall enough!

        for (int j = 0; j < numLines; ++j) {
            // clear screen
            printf("\033[A\33[2K\r");
        }

        float* guess = getOutput(ffnn);
        float cost = quadraticCost(guess, outputs[i], 10);

        char* imgStr = getImgStr(inputs[i], width, height);
        puts(imgStr);
        puts("       0123456789");
        printf("Label: ");

        int goodIndex = -1;
        for (int j = 0; j < 10; ++j) {
            if (outputs[i][j]==1) {
                goodIndex = j;
            }
            putchar(shade(outputs[i][j]));
        }
        putchar('\n');
        printf("Probs: ");
        float max = 0;
        int guessIndex = -1;
        for (int j = 0; j < 10; ++j) {
            if (guess[j] > max) {
                max = guess[j];
                guessIndex = j;
            }
            putchar(shade(guess[j]));
        }
        printf("  %d of %d\n", (i + 1), trainingSetSize);
        printf("Final: ");
        for (int j = 0; j < 10; ++j) {
            putchar(shade(j==guessIndex ? 1 : 0));
        }

        if (guessIndex == goodIndex) {
            puts("\033[0;32m  GOOD\033[0m");
        } else {
            puts("\033[0;31m  BAD!\033[0m");
        }
        
        putchar('\n');
        printf("Cost: %.3f\n", cost);

        freeNodes(gradient, ffnn->numLayers, ffnn->layerSizes);
    }
    putchar('\a');
}
