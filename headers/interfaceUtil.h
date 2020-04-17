#ifndef ASCII_PIXEL_H
#define ASCII_PIXEL_H

#include "FFNN.h"

enum ColorPairState {PAIR_DEF = 1, PAIR_GOOD, PAIR_BAD};
enum ProgramState {TRAINING, POST_TRAINING, TESTING};
enum ScreenState {SCREEN_BLANK, SCREEN_SET, SCREEN_NET};

struct DisplayState {
	enum ScreenState screenState;
	int outputIndex; // receptive field of output node at this index
	int shownIndex; // index displayed on screen within the set
};

int yCursor; // yCursor is set as needed,
int xCursor; // xCursor should be kept constant, and reset if changed

#define setCursor() move(yCursor, xCursor)
#define printNextLine(str) ++yCursor;setCursor();addstr(str)

void ncursesConfig();

/**
 * Prints the instructions, aligned to the bottom left of the screen
 */
void printInstructions();

/**
 * Clears the instructions aligned with the bottom right.
 */
void clearInstructions();

void clearTopLeftText(int numLines);

void handleUserInputAndTrain(struct DisplayState* ds, float** imgs, int numImages,
										int width, int height, float** labels, struct FFNN* ffnn);

void test(struct FFNN* ffnn, float** inputs, int width, int height, float** outputs, int setSize);

/**
 * Returns TRUE is training should be reset.
 */
bool handlePostTrainingInput(struct DisplayState* ds, float** imgs, int numImages,
								int width, int height, float** labels, struct FFNN* ffnn);

#endif
