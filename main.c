#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include "FFNN.h"
#include "MNISTRead.h"

#define DEMOTYPE 1 // set this to 1 or 0

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
    if (!DEMOTYPE) {
        puts("============Training Set============");
    }
    float** imgs = readMNISTTrainingImages(&numImages, &width, &height);
    float** labels = readMNISTTrainingLabels(&numImages);
    
    int inputLayerSize = width * height;
    int layerSizes[3] = {inputLayerSize, 
                        32,
                        10};
    struct FFNN* ffnn = allocFFNN(3, layerSizes);
    randomize(ffnn);
    setClassifier(ffnn);

    if (DEMOTYPE) {
        MNISTVisualSGD(ffnn, imgs, labels, numImages, 0.01f, width, height);
    } else {
        SGD(ffnn, imgs, labels, numImages, 0.01f);
        test(ffnn, imgs, labels, numImages);
        freeSet(imgs, labels, numImages);
        puts("==============Test Set==============");
        imgs = readMNISTTestImages(&numImages, &width, &height);
        labels = readMNISTTestLabels(&numImages);
        test(ffnn, imgs, labels, numImages);

    }
    freeFFNN(ffnn);
    freeSet(imgs, labels, numImages);

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
    int numLines = height + 5;
    for (int i = 0; i < numLines; ++i) {
        putchar('\n');
    }
    
    for (int i = 0; i < trainingSetSize; ++i) {
        setInput(ffnn, inputs[i]);
        forwardPass(ffnn);
        struct Node** gradient = backwardPass(ffnn, outputs[i]);
        applyGradient(ffnn, gradient, learningRate);
        // gradient freed below

        // =========
        // Everything after this is a visual addition to the SGD function in FFNN.c
        // Make sure to have the terminal window tall enough!

        for (int j = 0; j < numLines; ++j) {
            // clear screen
            printf("\033[A\33[2K\r");
        }

        float* guess = getOutput(ffnn);

        char* imgStr = getImgStr(inputs[i], width, height);
        puts(imgStr);
        puts("       0123456789");
        printf("Label: ");

        int goodIndex = -1;
        for (int j = 0; j < 10; ++j) {
            if (outputs[i][j]) {
                goodIndex = j;
            }
            putchar(shade(outputs[i][j]));
        }
        putchar('\n');
        printf("Probs: ");
        for (int j = 0; j < 10; ++j) {
            putchar(shade(guess[j]));
        }
        int guessIndex = maxIndex(guess, 10);
        printf("  %d of %d\n", (i + 1), trainingSetSize);
        printf("Final: ");
        for (int j = 0; j < 10; ++j) {
            putchar(shade(j==guessIndex ? guess[j] : 0));
        }

        if (guessIndex == goodIndex) {
            puts("\033[0;32m  GOOD\033[0m");
        } else {
            puts("\033[0;31m  BAD!\033[0m");
        }

        // assuming classifier for MNIST
        float cost = crossEntropyCost(guess, outputs[i], ffnn->layerSizes[ffnn->numLayers - 1]);
        printf("Cost: %.3f\n", cost);

        freeNodes(gradient, ffnn->numLayers, ffnn->layerSizes);
    }
    putchar('\a');
}
