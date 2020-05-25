#include <curses.h>
#include <stdlib.h>
#include <stdio.h>
#include "FFNN.h"
#include "interfaceUtil.h"
#include "MNISTRead.h"

int main(int argc, char **argv) {
	ncursesConfig();

	int numImages, width, height;
	yCursor = 1;
	printNextLine("Loading training set...");
	float** imgs = readMNISTTrainingImages(&numImages, &width, &height);
	float** labels = readMNISTTrainingLabels(&numImages);
	printNextLine("Ready.");
	printNextLine("Press space to start training.");

	int layerSizes[3] = {width * height, 
                        32,
                        10};
    struct FFNN* ffnn = allocFFNN(3, layerSizes);
    setClassifier(ffnn);

	bool flag = TRUE;
	struct DisplayState* ds = calloc(1, sizeof(struct DisplayState));
	while (flag) {
		randomize(ffnn);
		handleUserInputAndTrain(ds, imgs, numImages, width, height, labels, ffnn);
		clearTopLeftText(7);
		yCursor = 1;
		printNextLine("Training complete.");
		printNextLine("Press space to begin testing,");
		printNextLine("or backspace to restart.");
		flag = handlePostTrainingInput(ds, imgs, numImages, width, height, labels, ffnn);
	}
	free(ds);

	clearInstructions();
	clearTopLeftText(12);

	yCursor = 1;
	printNextLine("Loading testing set...");
	freeSet(imgs, labels, numImages);
	imgs = readMNISTTestImages(&numImages, &width, &height);
	labels = readMNISTTestLabels(&numImages);
	test(ffnn, imgs, width, height, labels, numImages);

	printNextLine("Testing complete. Press any key to exit.");
	flushinp();
	getch();
	freeSet(imgs, labels, numImages);
	freeFFNN(ffnn);

	endwin();
	return 0;
}
