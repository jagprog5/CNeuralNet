#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "FFNN.h"
#include "MNISTRead.h"

int main() {
    // TODO: create make file
    // gcc main.c FFNN.h FFNN.c MNISTRead.h MNISTRead.c -o ./a.exe -lm && ./a.exe

    int numImages;
    int width;
    int height;
    float** imgs = readMNISTImages(&numImages, &width, &height);
    float** labels = readMNISTLabels(&numImages);
    
    int inputLayerSize = width * height;
    int layers[5] = {inputLayerSize,
                        inputLayerSize >> 1, 
                        inputLayerSize >> 2, 
                        inputLayerSize >> 4, 
                        9};
    struct FFNN* ffnn = alloc(5, layers);

    stochasticTrain(ffnn, imgs, labels, numImages, 0.01f);

    // char* visualImg = getImgStr(imgs[0], width, height);
    // printf("%s", visualImg);

    return 0;
}