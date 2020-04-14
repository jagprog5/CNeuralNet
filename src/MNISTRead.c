#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <curses.h>
#include "MNISTRead.h"
#include "asciiPixel.h"

/**
 * Source for MNIST database files:
 * http://yann.lecun.com/exdb/mnist/
 */ 

void flipEndian32(uint32_t* in) {
	int b0, b1, b2, b3;
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

float** readMNISTImages(char* path, uint32_t* numImages, uint32_t* width, uint32_t* height) {
	FILE* fp = fopen(path, "rb");
	if (fp==NULL) {
		fprintf(stderr, "Error opening images file.\n");
		exit(1);
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
		set_cursor();
		printw("Reading Imgs: %d", i + 1);
		refresh();
	}

	fclose(fp);
	free(imgsBytes);
	return imgsOutput;
}

float** readMNISTLabels(char* path, uint32_t* numLabels) {
	FILE* fp = fopen(path, "rb");
	if (fp==NULL) {
		fprintf(stderr, "Error opiening images file.\n");
		exit(1);
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
		set_cursor();
		printw("Reading Labels: %d", i + 1);
		refresh();
	}

	fclose(fp);
	free(labels);
	return outputs;
}

float** readMNISTTrainingImages(uint32_t* numImages, uint32_t* width, uint32_t* height) {
	return readMNISTImages(TRAINING_IMAGES, numImages, width, height);
}

float** readMNISTTestImages(uint32_t* numImages, uint32_t* width, uint32_t* height) {
	return readMNISTImages(TEST_IMAGES, numImages, width, height);
}

float** readMNISTTrainingLabels(uint32_t* numLabels) {
	return readMNISTLabels(TRAINING_LABELS, numLabels);
}

float** readMNISTTestLabels(uint32_t* numLabels) {
	return readMNISTLabels(TEST_LABELS, numLabels);
}

void freeSet(float** inputs, float** outputs, int setSize) {
	for (int i = 0; i < setSize; ++i) {
		free(inputs[i]);
		free(outputs[i]);
	}
	free(inputs);
	free(outputs);
}

void printImg(float* MNISTImage, int width, int height) {
	int loc_y = 0;
	int loc_x;
	for (int j = 0; j < height; ++j) {
		loc_x = COLS - width - 1;
		++loc_y;
		move(loc_y, loc_x);
		for (int i = 0; i < width; ++i) {
			float pixel = 1;//MNISTImage[i + j * width];
			addch(shade(pixel));
		}
	}
	refresh();
}

