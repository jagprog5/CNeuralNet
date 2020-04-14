#include <curses.h>
#include <float.h>
#include "asciiPixel.h"

void initColorPairs() {
	init_pair(DEF_PAIR, COLOR_WHITE, COLOR_BLACK);
	init_pair(GOOD_PAIR, COLOR_GREEN, COLOR_BLACK);
	init_pair(BAD_PAIR, COLOR_RED, COLOR_BLACK);
}

char shade(float pixel) {
	char c;
	if (pixel < 0.1)		c = ' ';
		else if (pixel < 0.2)   c = '.';
		else if (pixel < 0.3)   c = ':';
		else if (pixel < 0.4)   c = '-';
		else if (pixel < 0.5)   c = '=';
		else if (pixel < 0.6)   c = '+';
		else if (pixel < 0.7)   c = '*';
		else if (pixel < 0.8)   c = '#';
		else if (pixel < 0.9)   c = '&';
		else			c = '$';
	return c;
}

void printImg(float* MNISTImage, int width, int height) {
	int locY = 0;
	int locX;
	for (int j = 0; j < height; ++j) {
		locX = COLS - width - 1;
		++locY;
		move(locY, locX);
		for (int i = 0; i < width; ++i) {
			float pixel = MNISTImage[i + j * width];
			addch(shade(pixel));
		}
	}
}

void printBlank(int width, int height) {
	int locY = 0;
	int locX;
	char arr[width];
	for (int i = 0; i < width; ++i) {
		arr[i] = shade(1);
	}
	arr[width] = '\0';
	for (int j = 0; j < height; ++j) {
		locX = COLS - width - 1;
		++locY;
		move(locY, locX);
		printw("%s", arr);

	}
}

/**
 * Returns index of largest magnitude value
 */
int maxMag(float* in, int num) {
    float max = -FLT_MAX;
    int index = -1;
    for (int i = 0; i < num; ++i) {
		float val = in[i] > 0 ? in[i] : in[i];
        if (val > max) {
            max = val;
            index = i;
        }
    }
    return index;
}

void printReceptiveField(float* field, int width, int height) {
    float max = field[maxMag(field, width * height)];
	int locY = 0;
	int locX;
	for (int j = 0; j < height; ++j) {
		locX = COLS - width - 1;
		++locY;
		move(locY, locX);
		for (int i = 0; i < width; ++i) {
			float val = field[i + j * width] / max;
			attron(COLOR_PAIR(val < 0 ? GOOD_PAIR : BAD_PAIR));
			addch(shade(val > 0 ? val : -val));
		}
	}
	attron(COLOR_PAIR(DEF_PAIR));
}