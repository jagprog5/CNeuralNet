#include <curses.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "FFNN.h"
#include "FFNNInspection.h"
#include "asciiPixel.h"
#include "MNISTRead.h"

#define SCREEN_BLANK 0
#define SCREEN_SET 1 // Shows dataset
#define SCREEN_NET 2 // Shows receptive field of network

#define addStrLine(str) addstr(str);++yCursor;setCursor();

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
	keypad(stdscr, TRUE);
	cbreak();
	noecho();
	nodelay(stdscr, FALSE);
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
	addstr("                            ");
	move(height + 1, COLS - width - 1);
	if (screenMode == SCREEN_BLANK) {
		addstr("      No output (fast)");
	} else if (screenMode == SCREEN_SET) {
		// center align variable sized string
		int halfLen = ((int)log10f(setIndex + 1) + 1 + 14) / 2;
		move(height + 1, COLS - width / 2 - halfLen - 1);
		printw("Dataset Image %d", setIndex + 1);
	} else if (screenMode == SCREEN_NET) {
		printw("  Receptive Field %d (slow)", outputIndex);
	}
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

void printInstructions() {
	int oldYCursor = yCursor;
	yCursor = LINES - 7;
	setCursor();
	attron(A_BOLD);
	addstr("Controls:");
	attrset(COLOR_PAIR(DEF_PAIR));
	++yCursor;
	setCursor();
	addstr("Use the LEFT and RIGHT arrow keys to toggle the screen.");
	++yCursor;
	setCursor();
	addstr("Use the UP and DOWN arrow keys to navigate the images.");
	++yCursor;
	setCursor();
	addstr("Press space to pause/resume.");
	++yCursor;
	setCursor();
	addstr("Press backspace to reset to the beginning of training.");
	yCursor = oldYCursor;
	setCursor();
}

void clearTopLeftText(int numLines) {
	yCursor = 2; // scroll back up to beginning to clear
	for (int i = 0; i < numLines; ++i) {
		setCursor();
		addstr("                                                        ");
		yCursor += 1;
	}
	yCursor -= numLines;
	setCursor();
}

void handleArrowInput(int ch, int *screenMode, bool *blankPulse,
						int *outputIndex, int* shownIndex, int numImages) {
	if (ch == KEY_RIGHT) {
		++*screenMode;
	} else if (ch == KEY_LEFT) {
		--*screenMode;
	} else if (ch == KEY_UP) {
		if (*screenMode == SCREEN_NET) {
			++*outputIndex;
		} else if (*screenMode == SCREEN_SET) {
			++*shownIndex;
		}
	} else if (ch == KEY_DOWN) {
		if (*screenMode == SCREEN_NET) {
			--*outputIndex;
		} else if (*screenMode == SCREEN_SET) {
			--*shownIndex;
		}
	}
	wrapRange(screenMode, 0, 3);
	if (*screenMode == SCREEN_BLANK) {
		*blankPulse = TRUE;
	}
	wrapRange(outputIndex, 0, 10);
	wrapRange(shownIndex, 0, numImages);
}

void handleOtherInput(int ch, bool *training, int *setIndex, int *shownIndex, struct FFNN* ffnn) {
	if (ch == ' ' || ch == KEY_BACKSPACE) {
		if (ch == ' ') {
			*training = !*training;
		} else if (ch == KEY_BACKSPACE) {
			*training = FALSE;
			*setIndex = 0;
			randomize(ffnn);
		}
		*shownIndex = *setIndex;
		nodelay(stdscr, training);
	}
}

void printProbs(bool empty, struct FFNN* ffnn, float** labels, int shownIndex) {
	float* outs = getOutput(ffnn);
	xCursor += 8;
	setCursor();
	addStrLine("0  1  2  3  4  5  6  7  8  9");
	if (empty) {
		addStrLine("-  -  -  -  -  -  -  -  -  -");
		addStrLine("-  -  -  -  -  -  -  -  -  -");
	} else {
		for (int i = 0; i < 10; ++i) {
			addch(shade(outs[i]));
			addstr("  ");
		}
		yCursor += 1;
		setCursor();
		float* label = labels[shownIndex];
		for (int i = 0; i < 10; ++i) {
			addch(shade(label[i]));
			addstr("  ");
		}
		yCursor += 1;
		setCursor();
	}
	xCursor -= 7;
	yCursor -= 3;
	setCursor();
	if (!empty) {
		int guessIndex = maxIndex(outs, 10);
		int goodIndex = maxIndex(labels[shownIndex], 10);
		if (guessIndex == goodIndex) {
			attron(COLOR_PAIR(GOOD_PAIR));
			addStrLine(" GOOD");
		} else {
			attron(COLOR_PAIR(BAD_PAIR));
			addStrLine(" BAD ");
		}
		attron(COLOR_PAIR(DEF_PAIR));
	} else {
		addStrLine(" ----");
	}

	addStrLine("Probs:");
	addStrLine("Label:");
	xCursor -= 1;
}

void handleUserInputAndTrain(float** imgs, int numImages, int width, int height,
								float** labels, struct FFNN* ffnn) {
	printInstructions();
	bool training = FALSE;
	bool blankPulse = TRUE; // update pulse, since BLANK doesn't print anything otherwise
	int screenMode = 0;
	int outputIndex = 0; // receptive field of output node at this index
	int setIndex = 0; // index in training set
	int shownIndex = 0; // index displayed on screen
	updateSideScreen(screenMode, outputIndex, shownIndex, imgs, width, height, ffnn);

	yCursor = 7;
	setCursor();
	addstr("Press space to start training.");
	refresh();

	while (setIndex < numImages) {
		int ch = getch();
		if (ch != ERR) {
			handleArrowInput(ch, &screenMode, &blankPulse, &outputIndex, &shownIndex, numImages);
			handleOtherInput(ch, &training, &setIndex, &shownIndex, ffnn);
		}

		if (training) {
			setInput(ffnn, imgs[shownIndex]);
			forwardPass(ffnn);
			struct Node** gradient = backwardPass(ffnn, labels[shownIndex]);
			applyGradient(ffnn, gradient, 0.01f);
			freeNodes(gradient, ffnn->numLayers, ffnn->layerSizes);
		} else if (blankPulse || screenMode != SCREEN_BLANK || ch == KEY_UP || ch == KEY_DOWN) {
			setInput(ffnn, imgs[shownIndex]);
			forwardPass(ffnn);
		}

		if (ch == ' ' || ch == KEY_BACKSPACE) {
			yCursor = 7;
			setCursor();
			if (training) {
				addstr("Press space to pause.           ");
			} else {
				addstr("Press space to resume.");
			}
		}

		yCursor = 8;
		setCursor();
		if (screenMode != SCREEN_BLANK) {
			printw("Training: %d       ", setIndex + 1);
		} else if (blankPulse) {
			addstr("Training...        ");
		}

		if (blankPulse || screenMode != SCREEN_BLANK) {
			yCursor = 10;
			setCursor();
			printProbs(blankPulse, ffnn, labels, shownIndex);
			updateSideScreen(screenMode, outputIndex, shownIndex, imgs, width, height, ffnn);
		}


		if (!training && ch == KEY_UP || ch == KEY_DOWN || ch == KEY_LEFT || ch == KEY_RIGHT) {
			updateSideScreen(screenMode, outputIndex, shownIndex, imgs, width, height, ffnn);
		}

		if (training) {
			++setIndex;
			shownIndex = setIndex;
		}
		if (blankPulse) {
			// pulse only lasts one frame
			blankPulse = FALSE;
		}
	}
}

void menu() {
	initNcursesWindow();
	int numImages, width, height;
	addStrLine("Loading training set...");
	float** imgs = readMNISTTrainingImages(&numImages, &width, &height);
	float** labels = readMNISTTrainingLabels(&numImages);
	addStrLine("Ready.");

	struct FFNN* ffnn = initNN(width * height);
	handleUserInputAndTrain(imgs, numImages, width, height, labels, ffnn);

	// // after training TODO

	nodelay(stdscr, FALSE);
	getch();
	endwin();
}