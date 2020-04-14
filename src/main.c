#include <curses.h>
#include <stdlib.h>
#include <stdio.h>
#include "FFNN.h"
#include "FFNNInspection.h"
#include "asciiPixel.h"
#include "MNISTRead.h"

#define SCREEN_BLANK 0
#define SCREEN_SET 1 // Shows dataset
#define SCREEN_NET 2 // Shows receptive field of network

void menu();

int main(int argc, char **argv) {
	menu();
}

void initNcursesWindow() {
	if (initscr() == NULL) {
		fprintf(stderr, "Error initialising ncurses.\n");
		exit(1);
	}
	start_color();
	initColorPairs();
	keypad(stdscr, 1);
	cbreak();
	noecho();
	nodelay(stdscr, 0);
	border(0, 0, 0, 0, 0, 0, 0, 0);
	yCursor = 2;
	xCursor = 3;
	setCursor();
}

struct FFNN* initNN(int numInputs) {
	int layerSizes[3] = {numInputs, 
                        32,
                        10};
    struct FFNN* ffnn = allocFFNN(3, layerSizes);
    randomize(ffnn);
    setClassifier(ffnn);
	return ffnn;
}

void updateSideScreen(int screenMode, int outputIndex, int setIndex, 
						float** imgs, int width, int height, struct FFNN* ffnn) {
	if (screenMode == SCREEN_BLANK) {
		printBlank(width, height);
	} else if (screenMode == SCREEN_SET) {
		printImg(imgs[setIndex], width, height);
	} else if (screenMode == SCREEN_NET) {
		float* field = malloc(sizeof(*field) * width * height);
		populateOutputReceptiveField(field, outputIndex, ffnn);
		printReceptiveField(field, width, height);
		free(field);
	}
	// Place text below screen
	move(height + 1, COLS - width - 1);
	if (screenMode == SCREEN_BLANK) {
		addstr("No output                  ");
	} else if (screenMode == SCREEN_SET) {
		printw("Dataset image %d    ", setIndex + 1);
	} else if (screenMode == SCREEN_NET) {
		printw("Receptive field of output %d", outputIndex);
	}
	refresh();
}

/**
 * wraps val to [min, max)
 */
void wrapRange(int* val, int min, int max) {
	if (*val < min) {
		*val = max - 1;
	} else if (*val > max - 1) {
		*val = min;
	}
}

void handleUserInputAndTrain(float** imgs, int numImages, int width, int height,
								float** labels, struct FFNN* ffnn) {
	yCursor = 2;
	setCursor();
	addstr("Use the LEFT and RIGHT arrow keys to toggle the screen.");
	++yCursor;
	setCursor();
	addstr("Use the UP and DOWN arrow keys to navigate the images.");
	++yCursor;
	setCursor();
	addstr("Press space to begin training.");
	refresh();

	int training = 0;
	int firstStart = 1;
	int screenMode = 0;
	int firstFrameOnBlank = 1;
	int outputIndex = 0;
	int setIndex = 0;
	while (setIndex < numImages) {
		int ch = getch();
		if (ch != ERR) {
			if (' ') {
				training = !training;
				if (firstStart) {
					firstStart = 0;
					setIndex = 0;
				}
				int numClear = 2 - yCursor; // number of lines to clear
				yCursor = 2;
				for (int i = 0; i < numClear; ++i) {
					setCursor();
					addstr("                                                        ");
					yCursor += 1;
				}
				yCursor -= numClear;
				setCursor();
				if (training) {
					addstr("Press space to pause.");
				} else {
					addstr("Press space to resume.");
				}
				++yCursor;
				nodelay(stdscr, training); // getch non blocking
			} else if (ch == KEY_RIGHT) {
				++screenMode;
			} else if (ch == KEY_LEFT) {
				--screenMode;
			} else if (screenMode == SCREEN_NET) {
				if (ch == KEY_UP) {
					++outputIndex;
				} else if (ch == KEY_DOWN) {
					--outputIndex;
				}
			} else if (screenMode == SCREEN_SET) {
				if (ch == KEY_UP) {
					++setIndex;
				} else if (ch == KEY_DOWN) {
					--setIndex;
				}
			}
			if (ch == KEY_UP || ch == KEY_DOWN || ch == KEY_LEFT || ch == KEY_RIGHT) {
				wrapRange(screenMode, 0, 3);
				if (screenMode == SCREEN_BLANK) {
					firstFrameOnBlank = 1;
				}
				wrapRange(outputIndex, 0, 10);
				wrapRange(setIndex, 0, numImages);
				if (!training) {
					updateSideScreen(screenMode, outputIndex, setIndex, imgs, width, height, ffnn);
				}
			}
		}
		if (training) {
			setCursor();
			if (firstFrameOnBlank) {
				firstFrameOnBlank = 0;
				addstr("Training...        ");
			} else {
				printw("Training: %d       ", setIndex + 1);
			}
			setInput(ffnn, imgs[setIndex]);
			forwardPass(ffnn);
			struct Node** gradient = backwardPass(ffnn, labels[setIndex]);
			applyGradient(ffnn, gradient, 0.01f);
			freeNodes(gradient, ffnn->numLayers, ffnn->layerSizes);

			float* guess = getOutput(ffnn);
			float loss = crossEntropyLoss(guess,
								labels[setIndex], ffnn->layerSizes[ffnn->numLayers - 1]);
			if (screenMode != SCREEN_BLANK || firstFrameOnBlank) {
				updateSideScreen(screenMode, outputIndex, setIndex, imgs, width, height, ffnn);
			}
		}
	}
}

void train(int *screenMode, int *outputIndex, int *setIndex, float** imgs,
			int width, int height, float** labels, int numImages, struct FFNN* ffnn) {
	int numClear = 3; // number of lines to clear
	yCursor = 2;
	for (int i = 0; i < numClear; ++i) {
		setCursor();
		addstr("                                                        ");
		yCursor += 1;
	}
	yCursor -= numClear;
	setCursor();
	addstr("Press space to pause.");
	++yCursor;

	for (int i = *setIndex; i < numImages; ++i) {
		int ch = getch();
		if (ch != ERR) {
			if (' ') {
				*setIndex = i;
				yCursor = 2;
				setCursor();
				addstr("Paused. Press space to resume.              ");
				return;
			} else if (ch == KEY_RIGHT || ch == KEY_LEFT) {
				if (ch == KEY_RIGHT) {
					++*screenMode;
					if (*screenMode > 2) {
						*screenMode = 0;
					}
				} else {
					--*screenMode;
					if (*screenMode < 0) {
						*screenMode = 2;
					}
				}
				if (*screenMode == SCREEN_BLANK) {
					setCursor();
					addstr("Training....    ");
					updateSideScreen(*screenMode, *outputIndex, i, imgs, width, height, ffnn);
				}
			} else if (*screenMode == SCREEN_NET) {
				if (ch == KEY_UP) {
					++*outputIndex;
					if (*outputIndex > 9) {
						*outputIndex = 0;
					}
				} else if (ch == KEY_DOWN) {
					--*outputIndex;
					if (*outputIndex < 0) {
						*outputIndex = 9;
					}
				}
			}
		}

		if (*screenMode != SCREEN_BLANK) {
			setCursor();
			printw("Training: %d       ", i + 1);
		}

        setInput(ffnn, imgs[i]);
        forwardPass(ffnn);
        struct Node** gradient = backwardPass(ffnn, labels[i]);
        applyGradient(ffnn, gradient, 0.01f);
        freeNodes(gradient, ffnn->numLayers, ffnn->layerSizes);

        float* guess = getOutput(ffnn);
        float loss = crossEntropyLoss(guess, labels[i], ffnn->layerSizes[ffnn->numLayers - 1]);

		if (*screenMode != SCREEN_BLANK) {
			updateSideScreen(*screenMode, *outputIndex, i, imgs, width, height, ffnn);
		}
		refresh();
    }
	*setIndex = -1;
}

void menu() {
	initNcursesWindow();
	int numImages, width, height;
	float** imgs = readMNISTTrainingImages(&numImages, &width, &height);
	++yCursor;
	float** labels = readMNISTTrainingLabels(&numImages);
	++yCursor;

	struct FFNN* ffnn = initNN(width * height);

	// while (setIndex != -1) {
	handleUserInputAndTrain(imgs, numImages, width, height, labels, ffnn);
		
		// train(&screenMode, &outputIndex, &setIndex, imgs, width, height, labels, numImages, ffnn);
	// }

	// after training TODO

	nodelay(stdscr, 0);
	getch();
	endwin();
}