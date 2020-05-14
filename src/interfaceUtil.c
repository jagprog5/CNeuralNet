#include <curses.h>
#include <float.h>
#include <stdlib.h>
#include <math.h>
#include "FFNN.h"
#include "FFNNInspection.h"
#include "interfaceUtil.h"

void ncursesConfig() {
	if (initscr() == NULL) {
		fprintf(stderr, "Error initialising ncurses.\n");
		exit(1);
	}
	int col, row;
	getmaxyx(stdscr,row,col);
	if (col < MINCOLS || row < MINROWS) {
		// TODO handle resizing window
		endwin();
		fprintf(stderr, "Increase the terminal size! Needs at least (%d, %d)\n", MINROWS, MINCOLS);
		exit(1);
	}
	start_color();
	init_pair(PAIR_DEF, COLOR_WHITE, COLOR_BLACK);
	init_pair(PAIR_GOOD, COLOR_GREEN, COLOR_BLACK);
	init_pair(PAIR_BAD, COLOR_RED, COLOR_BLACK);
	keypad(stdscr, TRUE); // enable arrow keys
	cbreak();	// pass keys directly from input without waiting for newline
	noecho();	// disable echo back of keys entered
	border(0, 0, 0, 0, 0, 0, 0, 0);
	xCursor = 3;
}

void printInstructions() {
	yCursor = LINES - 7;
	setCursor();
	attron(A_BOLD);
	addstr("Controls:");
	attrset(COLOR_PAIR(PAIR_DEF));
	printNextLine("Use the LEFT and RIGHT arrow keys to toggle the screen.");
	printNextLine("Use the UP and DOWN arrow keys to navigate the images.");
	printNextLine("Press space to pause/resume.");
	++yCursor;
	setCursor();
	#ifndef DOCKER
		static const char back[] = "backspace";
	#else
		static char back[] = "'p'";
	#endif
	printw("Press %s to reset to the beginning of training.", back);
}

void clearInstructions() {
	yCursor = LINES - 8;
	for (int i = 0; i < 5; ++i) {
		printNextLine("                                                        ");
	}
}

char shade(float pixel) {
	char c;
	if (pixel < 0.1)			c = ' ';
		else if (pixel < 0.2f)   c = '.';
		else if (pixel < 0.3f)   c = ':';
		else if (pixel < 0.4f)   c = '-';
		else if (pixel < 0.5f)   c = '=';
		else if (pixel < 0.6f)   c = '+';
		else if (pixel < 0.7f)   c = '*';
		else if (pixel < 0.8f)   c = '#';
		else if (pixel < 0.9f)   c = '&';
		else					c = '$';
	return c;
}

static void printImg(float* MNISTImage, int width, int height) {
	int locY = 0;
	int locX;
	for (int j = 0; j < height; ++j) {
		locX = COLS - width - 2;
		++locY;
		move(locY, locX);
		for (int i = 0; i < width; ++i) {
			float pixel = MNISTImage[i + j * width];
			addch(shade(pixel));
		}
	}
}

static void printBlank(int width, int height) {
	attron(A_DIM);
	int locY = 0;
	int locX;
	char arr[width + 1];
	for (int i = 0; i < width; ++i) {
		arr[i] = shade(1);
	}
	arr[width] = '\0';
	for (int j = 0; j < height; ++j) {
		++locY;
		locX = COLS - width - 2;
		move(locY, locX);
		printw("%s", arr);
	}
	attroff(A_DIM);
}

/**
 * Returns index of largest magnitude value
 */
static int maxMag(float* in, int num) {
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

static void printReceptiveField(float* field, int width, int height) {
	attron(A_BOLD);
    float max = field[maxMag(field, width * height)];
	int locY = 0;
	int locX;
	for (int j = 0; j < height; ++j) {
		locX = COLS - width - 2;
		++locY;
		move(locY, locX);
		for (int i = 0; i < width; ++i) {
			float val = field[i + j * width] / max;
			attron(COLOR_PAIR(val < 0 ? PAIR_GOOD : PAIR_BAD));
			addch(shade(val > 0 ? val : -val));
		}
	}
	attroff(A_BOLD);
	attron(COLOR_PAIR(PAIR_DEF));
}

static void printSideScreen(struct DisplayState *ds, float** imgs,
								int width, int height, struct FFNN* ffnn) {
	if (ds->screenState == SCREEN_BLANK) {
		printBlank(width, height);
	} else if (ds->screenState == SCREEN_SET) {
		printImg(imgs[ds->shownIndex], width, height);
	} else if (ds->screenState == SCREEN_NET) {
		float* field = malloc(sizeof(*field) * width * height);
		populateOutputReceptiveField(field, ds->outputIndex, ffnn);
		printReceptiveField(field, width, height);
		free(field);
	}
	// Place text below screen
	move(height + 1, COLS - width - 2);
	addstr("                            ");
	move(height + 1, COLS - width - 2);
	if (ds->screenState == SCREEN_BLANK) {
		addstr("      No output (fast)");
	} else if (ds->screenState == SCREEN_SET) {
		// center align variable sized string
		int halfLen = ((int)log10f(ds->shownIndex + 1) + 1 + 14) / 2;
		move(height + 1, COLS - width / 2 - halfLen - 3);
		printw("Dataset Image %d", ds->shownIndex + 1);
	} else if (ds->screenState == SCREEN_NET) {
		printw("     Receptive Field %d", ds->outputIndex);
	}
	refresh();
}

/**
 * ForwardPass the ffnn before calling this function
 * Aligned to the center left
 */
static void printProbs(bool empty, struct FFNN* ffnn, float** labels, int shownIndex) {
	float* outs;
	yCursor = 10;
	xCursor += 8; // make sure to reset xCursor by the end of funct
	setCursor();
	addstr("0  1  2  3  4  5  6  7  8  9");
	if (empty) {
		printNextLine("-  -  -  -  -  -  -  -  -  -");
		printNextLine("-  -  -  -  -  -  -  -  -  -");
	} else {
		outs = getOutput(ffnn);
		yCursor += 1;
		setCursor();
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
	}
	xCursor -= 7;
	yCursor -= 3;
	if (!empty) {
		int guessIndex = maxIndex(outs, 10);
		int goodIndex = maxIndex(labels[shownIndex], 10);
		if (guessIndex == goodIndex) {
			attron(COLOR_PAIR(PAIR_GOOD));
			printNextLine(" GOOD");
		} else {
			attron(COLOR_PAIR(PAIR_BAD));
			printNextLine(" BAD ");
		}
		attron(COLOR_PAIR(PAIR_DEF));
	} else {
		printNextLine(" ----");
	}
	printNextLine("Probs:");
	printNextLine("Label:");
	xCursor -= 1;
}

void clearTopLeftText(int numLines) {
	yCursor = 1; // scroll back up to beginning to clear
	for (int i = 0; i < numLines; ++i) {
		printNextLine("                                                        ");
	}
	yCursor -= numLines;
}

/**
 * wraps val to [min, max)
 */
static void wrapRange(int* val, int min, int max) {
	if (*val < min) {
		*val = max - 1;
	} else if (*val > max - 1) {
		*val = min;
	}
}

/**
 * *blankPulse can be NULL if not used
 */
static void handleArrowInput(int ch, struct DisplayState *ds, int numImages, bool *blankPulse) {
	if (ch == KEY_RIGHT) {
		++ds->screenState;
	} else if (ch == KEY_LEFT) {
		--ds->screenState;
	} else if (ch == KEY_UP) {
		if (ds->screenState == SCREEN_NET) {
			++ds->outputIndex;
		} else if (ds->screenState == SCREEN_SET) {
			++ds->shownIndex;
		}
	} else if (ch == KEY_DOWN) {
		if (ds->screenState == SCREEN_NET) {
			--ds->outputIndex;
		} else if (ds->screenState == SCREEN_SET) {
			--ds->shownIndex;
		}
	}
	wrapRange((int*)&ds->screenState, 0, 3);
	wrapRange((int*)&ds->outputIndex, 0, 10);
	wrapRange((int*)&ds->shownIndex, 0, numImages);
	if (blankPulse && ds->screenState == SCREEN_BLANK) {
		*blankPulse = TRUE;
	}
}

static void handleOtherInput(int ch, bool *training, int *setIndex, int *shownIndex, struct FFNN* ffnn) {
	if (ch == ' ' || ch == MY_BACKSPACE) {
		if (ch == ' ') {
			*training = !*training;
		} else if (ch == MY_BACKSPACE) {
			*training = FALSE;
			*setIndex = 0;
			randomize(ffnn);
		}
		*shownIndex = *setIndex;
		nodelay(stdscr, training);
	}
}

void handleUserInputAndTrain(struct DisplayState* ds, float** imgs, int numImages,
									int width, int height, float** labels, struct FFNN* ffnn) {
	printInstructions();
	printProbs(TRUE, NULL, NULL, 0);
	bool training = FALSE;
	bool blankPulse = TRUE; // draw pulse, since BLANK doesn't print anything otherwise
	int setIndex = 0; // Index in training set (as opposed to shownIndex in ds)
	printSideScreen(ds, imgs, width, height, ffnn);

	while (setIndex < numImages) {
		int ch = getch();
		if (ch != ERR) {
			handleArrowInput(ch, ds, numImages, &blankPulse);
			handleOtherInput(ch, &training, &setIndex, &ds->shownIndex, ffnn);
		}

		if (training) {
			setInput(ffnn, imgs[setIndex]);
			forwardPass(ffnn);
			struct Node** gradient = backwardPass(ffnn, labels[setIndex]);
			applyGradient(ffnn, gradient, 0.01f);
			freeNodes(gradient, ffnn->numLayers, ffnn->layerSizes);
		} else if (blankPulse || ds->screenState != SCREEN_BLANK || ch == KEY_UP || ch == KEY_DOWN) {
			// update NN outputs when paused and stepping through set
			setInput(ffnn, imgs[ds->shownIndex]);
			forwardPass(ffnn);
		}

		if (ch == ' ' || ch == MY_BACKSPACE) {
			yCursor = 6;
			setCursor();
			if (training) {
				addstr("Press space to pause.           ");
			} else {
				addstr("Press space to resume.");
			}
		}

		yCursor = 8;
		setCursor();
		if (ds->screenState != SCREEN_BLANK) {
			printw("Training: %d       ", setIndex + 1);
		} else if (blankPulse) {
			addstr("                   ");
		}

		if (blankPulse || ds->screenState != SCREEN_BLANK) {
			yCursor = 10;
			setCursor();
			printProbs(blankPulse, ffnn, labels, ds->shownIndex);
			printSideScreen(ds, imgs, width, height, ffnn);
		}

		if (!training && (ch == KEY_UP || ch == KEY_DOWN || ch == KEY_LEFT || ch == KEY_RIGHT)) {
			printSideScreen(ds, imgs, width, height, ffnn);
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
}

void test(struct FFNN* ffnn, float** inputs, int width, int height, float** outputs, int setSize) {
    int errorCount = 0;
    int numOutputs = ffnn->layerSizes[ffnn->numLayers - 1];
	struct DisplayState ds;
	ds.screenState = SCREEN_SET;
    for (int i = 0; i < setSize; ++i) {
		ds.shownIndex = i;
        setInput(ffnn, inputs[i]);
        forwardPass(ffnn);
        int guessIndex = maxIndex(getOutput(ffnn), numOutputs);
        int goodIndex = maxIndex(outputs[i], numOutputs);
        if (guessIndex != goodIndex) {
			attron(A_DIM);
            attron(COLOR_PAIR(PAIR_BAD));
            errorCount += 1;
        }
		yCursor = 7;
		setCursor();
        printw("Error Rate: %.2f%% (%d)  ", 100 * (float)errorCount / (i + 1), i + 1);
		refresh();
		if (guessIndex != goodIndex) {
			attroff(A_DIM);
            attron(COLOR_PAIR(PAIR_DEF));
        }
		printSideScreen(&ds, inputs, width, height, ffnn);
    }
}

bool handlePostTrainingInput(struct DisplayState* ds, float** imgs, int numImages,
								int width, int height, float** labels, struct FFNN* ffnn) {
	ds->shownIndex = numImages - 1;
	while (1) {
		int ch = getch();
		handleArrowInput(ch, ds, numImages, NULL);
		if (ch == MY_BACKSPACE) {
			clearTopLeftText(12);
			yCursor = 5;
			printNextLine("Training reset.");
			return TRUE;
		} else if (ch == ' ') {
			return FALSE;
		} else if (ch == KEY_UP || ch == KEY_DOWN || ch == KEY_LEFT || ch == KEY_RIGHT) {
			printSideScreen(ds, imgs, width, height, ffnn);
			if (ds->screenState != SCREEN_BLANK) {
				setInput(ffnn, imgs[ds->shownIndex]);
				forwardPass(ffnn);
			}
			printProbs(ds->screenState == SCREEN_BLANK, ffnn, labels, ds->shownIndex);
		}
	}
}
