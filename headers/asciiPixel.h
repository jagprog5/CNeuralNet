#ifndef ASCII_PIXEL_H
#define ASCII_PIXEL_H

#define DEF_PAIR 1
#define GOOD_PAIR 2
#define BAD_PAIR 3

int yCursor;
int xCursor;

#define setCursor() move(yCursor, xCursor)

void initColorPairs();

char shade(float pixel);

void printImg(float* MNISTImage, int width, int height);

void printBlank(int width, int height);

void printReceptiveField(float* field, int width, int height);

#endif
