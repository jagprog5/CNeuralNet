#include <stdint.h>

#ifndef MNIST_READ_H
#define MNIST_READ_H

#define TRAINING_IMAGES "./train-images-idx3-ubyte"
#define TRAINING_LABELS "./train-labels-idx1-ubyte"
#define TEST_IMAGES "./t10k-images-idx3-ubyte"
#define TEST_LABELS "./t10k-labels-idx1-ubyte"

// assuming 4 byte int and float

float** readMNISTTrainingImages(uint32_t* numImages, uint32_t* width, uint32_t* height);

float** readMNISTTestImages(uint32_t* numImages, uint32_t* width, uint32_t* height);

float** readMNISTTrainingLabels(uint32_t* numLabels);

float** readMNISTTestLabels(uint32_t* numLabels);

char shade(float pixel);

char* getImgStr(float* MNISTImage, uint32_t width, uint32_t height);

#endif