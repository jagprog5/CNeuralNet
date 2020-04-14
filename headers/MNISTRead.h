#ifndef MNIST_READ_H
#define MNIST_READ_H

#define TRAINING_IMAGES "./reasources/train-images-idx3-ubyte"
#define TRAINING_LABELS "./reasources/train-labels-idx1-ubyte"
#define TEST_IMAGES "./reasources/t10k-images-idx3-ubyte"
#define TEST_LABELS "./reasources/t10k-labels-idx1-ubyte"

#include <stdint.h>

float** readMNISTTrainingImages(uint32_t* numImages, uint32_t* width, uint32_t* height);

float** readMNISTTestImages(uint32_t* numImages, uint32_t* width, uint32_t* height);

float** readMNISTTrainingLabels(uint32_t* numLabels);

float** readMNISTTestLabels(uint32_t* numLabels);

void freeSet(float** inputs, float** outputs, int setSize);

#endif
