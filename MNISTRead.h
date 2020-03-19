#include <stdint.h>

#ifndef MNIST_READ_H
#define MNIST_READ_H

#define TRAINING_IMAGES "./train-images-idx3-ubyte"
#define TRAINING_LABELS "./train-labels-idx1-ubyte"

// assuming 4 byte int and float

float** readMNISTImages(uint32_t* numImages, uint32_t* width, uint32_t* height);

float** readMNISTLabels(uint32_t* numLabels);

char shade(float pixel);

char* getImgStr(float* MNISTImage, uint32_t width, uint32_t height);

#endif