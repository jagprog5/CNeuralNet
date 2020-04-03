#include <stdio.h>
#include <stdlib.h>
#include "MNISTRead.h"
#include "asciiPixel.h"
#include "printReducer.h"
#include "FFNN.h"
#include "FFNNInspection.h"

// there's some copy and paste between demos, but it keeps it simple

void demoReduced() {
    int numImages;
    int width;
    int height;
    puts("============Training Set============");
    float** imgs = readMNISTTrainingImages(&numImages, &width, &height);
    float** labels = readMNISTTrainingLabels(&numImages);
    
    int layerSizes[3] = {width * height, 
                        32,
                        10};
    struct FFNN* ffnn = allocFFNN(3, layerSizes);
    randomize(ffnn);
    setClassifier(ffnn);

    SGD(ffnn, imgs, labels, numImages, 0.01f);
    test(ffnn, imgs, labels, numImages);
    freeSet(imgs, labels, numImages);

    puts("==============Test Set==============");
    imgs = readMNISTTestImages(&numImages, &width, &height);
    labels = readMNISTTestLabels(&numImages);
    test(ffnn, imgs, labels, numImages);
    freeSet(imgs, labels, numImages);
    freeFFNN(ffnn);
}

void demoVisual() {
    int numImages;
    int width;
    int height;
    puts("============Training Set============");
    float** imgs = readMNISTTrainingImages(&numImages, &width, &height);
    float** labels = readMNISTTrainingLabels(&numImages);
    
    int layerSizes[3] = {width * height, 
                        32,
                        10};
    struct FFNN* ffnn = allocFFNN(3, layerSizes);
    randomize(ffnn);
    setClassifier(ffnn);

    int numLines = height + 5;
    for (int i = 0; i < numLines; ++i) {
        putchar('\n');
    }

    for (int i = 0; i < numImages; ++i) {
        setInput(ffnn, imgs[i]);
        forwardPass(ffnn);
        struct Node** gradient = backwardPass(ffnn, labels[i]);
        applyGradient(ffnn, gradient, 0.01f);
        freeNodes(gradient, ffnn->numLayers, ffnn->layerSizes);

        // =========
        // Everything after this is a visual addition to the SGD function in FFNN.c
        // Make sure to have the terminal window tall enough!

        prt_redu(i + 1, 300,
        for (int j = 0; j < numLines; ++j) {
            // clear screen
            printf("\033[A\33[2K\r");
        }

        float* guess = getOutput(ffnn);
        char* imgStr = getImgStr(imgs[i], width, height);
        puts(imgStr);
        free(imgStr);
        printf("       0123456789\nLabel: ");
        int goodIndex = -1;
        for (int j = 0; j < 10; ++j) {
            if (labels[i][j]) {
                goodIndex = j;
            }
            putchar(shade(labels[i][j]));
        }
        printf("  %d of %d\nProbs: ", i + 1, numImages);
        for (int j = 0; j < 10; ++j) {
            putchar(shade(guess[j]));
        }
        
        // assuming classifier for MNIST
        float loss = crossEntropyLoss(guess, labels[i], ffnn->layerSizes[ffnn->numLayers - 1]);
        printf("  Loss: %.3f\nFinal: ", loss);
        int guessIndex = maxIndex(guess, 10);        
        for (int j = 0; j < 10; ++j) {
            putchar(shade(j==guessIndex ? guess[j] : 0));
        }

        if (guessIndex == goodIndex) {
            puts("\033[0;32m  GOOD\033[0m");
        } else {
            puts("\033[0;31m  BAD!\033[0m");
        })
    }
    putchar('\a');

    test(ffnn, imgs, labels, numImages);
    freeSet(imgs, labels, numImages);

    puts("==============Test Set==============");
    imgs = readMNISTTestImages(&numImages, &width, &height);
    labels = readMNISTTestLabels(&numImages);
    test(ffnn, imgs, labels, numImages);
    freeSet(imgs, labels, numImages);
    freeFFNN(ffnn);
}

void demoProgression(int nodeID) {
    int numImages;
    int width;
    int height;
    puts("============Training Set============");
    float** imgs = readMNISTTrainingImages(&numImages, &width, &height);
    float** labels = readMNISTTrainingLabels(&numImages);
    
    int layerSizes[3] = {width * height, 
                        32,
                        10};
    struct FFNN* ffnn = allocFFNN(3, layerSizes);
    randomize(ffnn);
    setClassifier(ffnn);

    float* receptiveField = malloc(sizeof(*receptiveField) * width * height);

    int numLines = height + 2;
    for (int i = 0; i < numLines; ++i) {
        putchar('\n');
    }

    for (int i = 0; i < numImages; ++i) {
        setInput(ffnn, imgs[i]);
        forwardPass(ffnn);
        struct Node** gradient = backwardPass(ffnn, labels[i]);
        applyGradient(ffnn, gradient, 0.01f);
        freeNodes(gradient, ffnn->numLayers, ffnn->layerSizes);

        prt_redu(i + 1, 300,
        if ((i + 1) % 10 == 0) {
            for (int j = 0; j < numLines; ++j) {
                // clear screen
                printf("\033[A\33[2K\r");
            }

            populateOutputReceptiveField(receptiveField, nodeID, ffnn);
            char* img = getReceptiveFieldImgStr(receptiveField, width, height);
            puts(img);
            free(img);
            printf("%d of %d\n", i + 1, numImages);
        }
        )
    }
    putchar('\a');

    freeSet(imgs, labels, numImages);
    free(receptiveField);
    freeFFNN(ffnn);
}