#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include "FFNN.h"
#include "MNISTRead.h"

int main() {
    // Hand-held this tutorial in double checking NN. Values are the same
    // https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
    int layers[3] = {2, 1, 2};
    float** vals = malloc(sizeof(float*) * 3);
    float hidden1[] = {0.763774618976614, 0.13436424411240122, 0.8474337369372327};
    vals[0] = hidden1;
    float hidden2[] = {0.49543508709194095, 0.2550690257394217, 0.651592972722763, 0.4494910647887381};
    vals[1] = hidden2;

    struct FFNN* ffnn = alloc(3, layers);
    setNetwork(ffnn, vals);
    print(ffnn);
    float in[] = {1, 0};
    setInput(ffnn, in);
    forwardPass(ffnn);
    float* out = getOutput(ffnn);
    printf("Output from forward pass: %f,%f\n", out[0], out[1]);
    
    float expected[] = {0, 1};
    struct NodeGradient** gradient = backwardPass(ffnn, expected);
    printf("%f\n", gradient[1][0].dBias);
    printf("%f,%f\n", gradient[2][0].dBias, gradient[2][1].dBias);

    applyGradient(ffnn, gradient, 1);
    print(ffnn);
    return 0;
}

// void MNISTVisualStochasticTrain(struct FFNN* ffnn,
//                     float** inputs, 
//                     float** outputs, 
//                     int trainingSetSize, 
//                     float learningRate,
//                     int width,
//                     int height);

// float sig(float in) {
//     return 1 / (1 + powf(M_E, -in));
// }


// int main() {
//     int numImages;
//     int width;
//     int height;
//     float** imgs = readMNISTImages(&numImages, &width, &height);
//     float** labels = readMNISTLabels(&numImages);
    
//     int inputLayerSize = width * height;
//     int layers[5] = {inputLayerSize, 
//                         inputLayerSize >> 2,
//                         inputLayerSize >> 4,
//                         inputLayerSize >> 6,
//                         10};
//     struct FFNN* ffnn = alloc(3, layers);
//     randomize(ffnn);
//     // stochasticTrain(ffnn, imgs, labels, numImages, 0.01f);
//     MNISTVisualStochasticTrain(ffnn, imgs, labels, numImages, 0.001f, width, height);
//     return 0;
// }


// void MNISTVisualStochasticTrain(struct FFNN* ffnn,
//                     float** inputs, 
//                     float** outputs, 
//                     int trainingSetSize, 
//                     float learningRate,
//                     int width,
//                     int height) {

//     puts("========Training========");
//     int numLines = height + 5;
//     for (int i = 0; i < numLines; ++i) {
//         putchar('\n');
//     }
    
//     for (int i = 0; i < trainingSetSize; ++i) {
//         // sleep(1);
//         setInput(ffnn, inputs[i]);
//         forwardPass(ffnn);
//         struct NodeGradient** gradient = backwardPass(ffnn, outputs[i]);
//         applyGradient(ffnn, gradient, learningRate);
//         free(gradient + 1);

//         // =========
//         // Make sure to have the terminal window tall enough


//         for (int j = 0; j < numLines; ++j) {
//             printf("\033[A\33[2K\r");
//         }

//         float* guess = getOutput(ffnn);
//         float cost = quadraticCost(guess, outputs[i], 10);

//         char* imgStr = getImgStr(inputs[i], width, height);
//         puts(imgStr);
//         puts("       0123456789");
//         printf("Label: ");
//         for (int j = 0; j < 10; ++j) {
//             putchar(shade(outputs[i][j]));
//         }
//         putchar('\n');
//         printf("Guess: ");
//         for (int j = 0; j < 10; ++j) {
//             putchar(shade(guess[j]));
//         }
//         putchar('\n');
//         printf("Cost: %.3f\n", cost);
//     }
//     putchar('\a');
// }
