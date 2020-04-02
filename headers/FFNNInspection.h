#ifndef FORWARD_NETWORK_INSPECTION_H
#define FORWARD_NETWORK_INSPECTION_H

#define CLEARLN 4
#define COLORLN 7

#include "FFNN.h"

void inspect(struct FFNN* ffnn);

int maxIndex(float* in, int num);

int minIndex(float* in, int num);

void test(struct FFNN* ffnn, float** inputs, float** outputs, int setSize);

float*** allocReceptiveFields(struct FFNN* ffnn);

void populateReceptiveFields(float*** fields, struct FFNN* ffnn);

void populateOutputReceptiveField(float* receptiveField, int outputNode, struct FFNN* ffnn);

char* getReceptiveFieldImgStr(float* img, int width, int height);

void freeReceptiveFields(float*** fields, struct FFNN* ffnn);

#endif