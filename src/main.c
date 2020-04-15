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

#define addStrLine(str) addstr(str);++yCursor;setCursor()

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

struct DisplayState {
	int screenMode;
	int outputIndex; // receptive field of output node at this index
	int shownIndex; // index displayed on screen within the set
};

void updateSideScreen(struct DisplayState *ds, float** imgs, int width, int height, struct FFNN* ffnn) {
	if (ds->screenMode == SCREEN_BLANK) {
		printBlank(width, height);
	} else if (ds->screenMode == SCREEN_SET) {
		printImg(imgs[ds->shownIndex], width, height);
	} else if (ds->screenMode == SCREEN_NET) {
		float* field = malloc(sizeof(*field) * width * height);
		populateOutputReceptiveField(field, ds->outputIndex, ffnn);
		printReceptiveField(field, width, height);
		free(field);
	}
	// Place text below screen
	move(height + 1, COLS - width - 1);
	addstr("                            ");
	move(height + 1, COLS - width - 1);
	if (ds->screenMode == SCREEN_BLANK) {
		addstr("      No output (fast)");
	} else if (ds->screenMode == SCREEN_SET) {
		// center align variable sized string
		int halfLen = ((int)log10f(ds->shownIndex + 1) + 1 + 14) / 2;
		move(height + 1, COLS - width / 2 - halfLen - 1);
		printw("Dataset Image %d", ds->shownIndex + 1);
	} else if (ds->screenMode == SCREEN_NET) {
		printw("      Receptive Field %d", ds->outputIndex);
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

void clearInstructions() {
	int oldYCursor = yCursor;
	yCursor = LINES - 7;
	for (int i = 0; i < 5; ++i) {
		setCursor();
		addstr("                                                        ");
		yCursor += 1;
	}
	yCursor -= 5;
	setCursor();
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

void handleArrowInput(int ch, struct DisplayState *ds, int numImages) {
	if (ch == KEY_RIGHT) {
		++ds->screenMode;
	} else if (ch == KEY_LEFT) {
		--ds->screenMode;
	} else if (ch == KEY_UP) {
		if (ds->screenMode == SCREEN_NET) {
			++ds->outputIndex;
		} else if (ds->screenMode == SCREEN_SET) {
			++ds->shownIndex;
		}
	} else if (ch == KEY_DOWN) {
		if (ds->screenMode == SCREEN_NET) {
			--ds->outputIndex;
		} else if (ds->screenMode == SCREEN_SET) {
			--ds->shownIndex;
		}
	}
	wrapRange(&ds->screenMode, 0, 3);
	wrapRange(&ds->outputIndex, 0, 10);
	wrapRange(&ds->shownIndex, 0, numImages);
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

struct DisplayState* handleUserInputAndTrain(float** imgs, int numImages, int width, int height,
								float** labels, struct FFNN* ffnn) {
	printInstructions();
	bool training = FALSE;
	bool blankPulse = TRUE; // draw pulse, since BLANK doesn't print anything otherwise
	struct DisplayState *ds = calloc(1, sizeof(struct DisplayState));
	int setIndex = 0; // Index in training set (as opposed to shownIndex in ds)
	updateSideScreen(ds, imgs, width, height, ffnn);

	while (setIndex < numImages) {
		int ch = getch();
		if (ch != ERR) {
			handleArrowInput(ch, ds, numImages);
			handleOtherInput(ch, &training, &setIndex, &ds->shownIndex, ffnn);
			if (ds->screenMode == SCREEN_BLANK) {
				blankPulse = TRUE;
			}
		}

		if (training) {
			setInput(ffnn, imgs[setIndex]);
			forwardPass(ffnn);
			struct Node** gradient = backwardPass(ffnn, labels[setIndex]);
			applyGradient(ffnn, gradient, 0.01f);
			freeNodes(gradient, ffnn->numLayers, ffnn->layerSizes);
		} else if (blankPulse || ds->screenMode != SCREEN_BLANK || ch == KEY_UP || ch == KEY_DOWN) {
			setInput(ffnn, imgs[ds->shownIndex]);
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
		if (ds->screenMode != SCREEN_BLANK) {
			printw("Training: %d       ", setIndex + 1);
		} else if (blankPulse) {
			addstr("Training...        ");
		}

		if (blankPulse || ds->screenMode != SCREEN_BLANK) {
			yCursor = 10;
			setCursor();
			printProbs(blankPulse, ffnn, labels, ds->shownIndex);
			updateSideScreen(ds, imgs, width, height, ffnn);
		}


		if (!training && ch == KEY_UP || ch == KEY_DOWN || ch == KEY_LEFT || ch == KEY_RIGHT) {
			updateSideScreen(ds, imgs, width, height, ffnn);
		}

		if (training) {
			++setIndex;
			ds->shownIndex = setIndex;
		}
		if (blankPulse) {
			// pulse only lasts one frame
			blankPulse = FALSE;
		}
	}
	nodelay(stdscr, FALSE);
	return ds;
}

/**
 * Returns TRUE is training should be reset.
 */
bool handlePostTrainingInput(struct DisplayState* ds, float** imgs, int numImages,
								int width, int height, float** labels, struct FFNN* ffnn) {
	ds->shownIndex = numImages - 1;
	while (1) {
		int ch = getch();
		handleArrowInput(ch, ds, numImages);
		if (ch == KEY_BACKSPACE) {
			clearTopLeftText(12);
			yCursor = 7;
			setCursor();
			addStrLine("Training reset.");
			return TRUE;
		} else if (ch == ' ') {
			return FALSE;
		} else if (ch == KEY_UP || ch == KEY_DOWN || ch == KEY_LEFT || ch == KEY_RIGHT) {
			updateSideScreen(ds, imgs, width, height, ffnn);

			if (ds->screenMode != SCREEN_BLANK) {
				setInput(ffnn, imgs[ds->shownIndex]);
				forwardPass(ffnn);
			}
			yCursor = 10;
			setCursor();
			printProbs(ds->screenMode == SCREEN_BLANK, ffnn, labels, ds->shownIndex);
		}
	}
}

void test(struct FFNN* ffnn, float** inputs, int width, int height, float** outputs, int setSize) {
    int errorCount = 0;
    int numOutputs = ffnn->layerSizes[ffnn->numLayers - 1];
	struct DisplayState ds;
	ds.screenMode = SCREEN_SET;
    for (int i = 0; i < setSize; ++i) {
		ds.shownIndex = i;
        setInput(ffnn, inputs[i]);
        forwardPass(ffnn);
        int guessIndex = maxIndex(getOutput(ffnn), numOutputs);
        int goodIndex = maxIndex(outputs[i], numOutputs);
        if (guessIndex != goodIndex) {
            attron(COLOR_PAIR(BAD_PAIR));
            errorCount += 1;
        }
		yCursor = 7;
		setCursor();
        printw("Error Rate: %.2f%% (%d)", 100 * (float)errorCount / (i + 1), i + 1);
		refresh();
		if (guessIndex != goodIndex) {
            attron(COLOR_PAIR(DEF_PAIR));
        }
		updateSideScreen(&ds, inputs, width, height, ffnn);
    }
    ++yCursor;
	setCursor();
}

void menu() {
	initNcursesWindow();
	int numImages, width, height;
	addStrLine("Loading training set...");
	float** imgs = readMNISTTrainingImages(&numImages, &width, &height);
	float** labels = readMNISTTrainingLabels(&numImages);
	addStrLine("Ready.");
	yCursor = 7;
	setCursor();
	addstr("Press space to start training.");

	struct FFNN* ffnn = initNN(width * height);

	bool flag = TRUE;
	struct DisplayState* ds;
	while (flag) {
		ds = handleUserInputAndTrain(imgs, numImages, width, height, labels, ffnn);

		clearTopLeftText(7);
		addStrLine("Training complete.");
		addStrLine("Press space to begin testing,");
		addStrLine("or backspace to restart.");

		flag = handlePostTrainingInput(ds, imgs, numImages, width, height, labels, ffnn);
		free(ds);
	}

	clearInstructions();
	clearTopLeftText(12);
	ds->screenMode = SCREEN_SET;
	ds->shownIndex = 0;

	yCursor = 2;
	setCursor();
	addStrLine("Loading testing set...");

	freeSet(imgs, labels, numImages);
	imgs = readMNISTTestImages(&numImages, &width, &height);
	labels = readMNISTTestLabels(&numImages);

	test(ffnn, imgs, width, height, labels, numImages);

	addStrLine("Testing complete. Press any key to exit.");
	flushinp();
	getch();

	freeSet(imgs, labels, numImages);
	freeFFNN(ffnn);

	endwin();
}