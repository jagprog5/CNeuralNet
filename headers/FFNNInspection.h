#ifndef FORWARD_NETWORK_INSPECTION_H
#define FORWARD_NETWORK_INSPECTION_H

#include "FFNN.h"

void inspect(struct FFNN* ffnn);

int maxIndex(float* in, int num);

void populateOutputReceptiveField(float* receptiveField, int outputNode, struct FFNN* ffnn);

#endif