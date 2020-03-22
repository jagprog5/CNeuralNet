#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "MNISTRead.h"

/**
 * Source for MNIST database files:
 * http://yann.lecun.com/exdb/mnist/
 */ 

void flipEndian32(uint32_t* in) {
    int b0,b1,b2,b3;
    b0 = *in & 0xFF;
    b1 = (*in >> 8) & 0xFF;
    b2 = (*in >> 16) & 0xFF;
    b3 = (*in >> 24) & 0xFF;
    *in = (b0 << 24) + (b1 << 16) + (b2 << 8) + b3;
}

void readFlipEndian32(FILE* fp, uint32_t* input) {
    fread(input, sizeof(*input), 1, fp);
    flipEndian32(input);
}

float** readMNISTImages(uint32_t* numImages, uint32_t* width, uint32_t* height) {
    FILE* fp = fopen(TRAINING_IMAGES, "rb");
    if (fp==NULL) {
        puts("Error opening file!");
        return NULL;
    }

    uint32_t magicNumber;
    readFlipEndian32(fp, &magicNumber);
    readFlipEndian32(fp, numImages);
    readFlipEndian32(fp, width);
    readFlipEndian32(fp, height);

    float** imgsOutput = malloc(sizeof(*imgsOutput) * *numImages);
    uint64_t imgsDataLen = *numImages * *width * *height;
    uint8_t* imgsBytes = malloc(sizeof(*imgsBytes) * imgsDataLen);
    fread(imgsBytes, sizeof(*imgsBytes), imgsDataLen, fp);

    for (uint32_t i = 0; i < *numImages; ++i) {
        uint32_t imgDataLen = *width * *height;
        imgsOutput[i] = malloc(sizeof(float) * imgDataLen);
        for (int j = 0; j < imgDataLen; ++j) {
            imgsOutput[i][j] = (float)imgsBytes[j + i * imgDataLen] / 0xFF;
        }
    }

    fclose(fp);
    free(imgsBytes);
    return imgsOutput;
}

char shade(float pixel) {
    char c;
    if (pixel < 0.1)            c = ' ';
        else if (pixel < 0.25)  c = '.';
        else if (pixel < 0.5)   c = ':';
        else if (pixel < 0.75)  c = '|';
        else if (pixel < 0.9)   c = 'X';
        else                    c = '#';
    return c;
}

char* getImgStr(float* MNISTImage, uint32_t width, uint32_t height) {
                                    // +height for each newline char

    char* out = malloc(sizeof(*out) * (width * height + height));

    int imgWalk = 0;
    int outWalk = 0;
    while (imgWalk < width * height) {
        float pixel = MNISTImage[imgWalk++];
        char c = shade(pixel);
        out[outWalk++] = c;

        if (imgWalk % width == 0) {
            out[outWalk++] = '\n';
        }
    }
    return out;
}

float** readMNISTLabels(uint32_t* numLabels) {
    FILE* fp = fopen(TRAINING_LABELS, "rb");
    if (fp==NULL) {
        puts("Error opening file!");
        return NULL;
    }

    uint32_t magicNumber;
    readFlipEndian32(fp, &magicNumber);
    readFlipEndian32(fp, numLabels);

    uint8_t* labels = malloc(sizeof(*labels) * *numLabels);
    fread(labels, sizeof(*labels), *numLabels, fp);

    float** outputs = malloc(sizeof(*outputs) * *numLabels);

    for (uint32_t i = 0; i < *numLabels; ++i) {
        outputs[i] = calloc(10, sizeof(float));
        outputs[i][labels[i]] = 1;
    }

    fclose(fp);
    free(labels);
    return outputs;
}