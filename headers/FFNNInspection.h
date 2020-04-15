#ifndef FORWARD_NETWORK_INSPECTION_H
#define FORWARD_NETWORK_INSPECTION_H

#include "FFNN.h"

void inspect(struct FFNN* ffnn);

void test(struct FFNN* ffnn, float** inputs, float** outputs, int setSize);

void populateOutputReceptiveField(float* receptiveField, int outputNode, struct FFNN* ffnn);

#endif