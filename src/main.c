#include <stdlib.h>
#include <stdio.h>
#include <curses.h>
#include "asciiPixel.h"
#include "MNISTRead.h"

#define SCREEN_BLANK 0
#define SCREEN_SET 1
#define SCREEN_NET 2



int main(int argc, char **argv) {
	int screenMode = 0;

	if (initscr() == NULL) {
		fprintf(stderr, "Error initialising ncurses.\n");
		exit(1);
	}
	start_color();
	init_color_pairs();
	cbreak();
	noecho();
	keypad(stdscr, TRUE);

	nodelay(stdscr, TRUE);
	border(0, 0, 0, 0, 0, 0, 0, 0);
	y_cursor = 2;
	x_cursor = 3;
	set_cursor();
	addstr("Press space to toggle screen output ->");
	++y;
	set_cursor();
	addstr("Press enter to load the training set.");
	while (1) {
		ch = getch();
		switch (ch) {
			case ' ':
				//
				break;
			case KEY_ENTER:
				//
				break;
			default:
				break;
		}
	}
	
	int numImages, width, height;
	//float** imgs = readMNISTTrainingImages(&numImages, &width, &height);
	//for (int i = 0; i < numImages; ++i) {
//		printImg(imgs[i], width, height);
//	}

	nodelay(stdscr, FALSE);
	getch();
	endwin();


	/*
	
	// 0 is reduced
	// 1 is visual
	// 2 is progression
	int demoType = 1;
	if (argc > 1) {
		demoType = argv[1][0] - '0';
		if (demoType < 0 || demoType > 2) {
			demoType = 0;
		}
	}
	int nodeID = 0;
	if (demoType == 2 && argc > 2) {
		nodeID = argv[2][0] - '0';
		if (nodeID < 0 || nodeID > 9) {
			nodeID = 0;
		}
	}

	switch (demoType) {
		case 0:
			demoReduced();
			break;
		case 1:
			demoVisual();
			break;
		case 2:
			demoProgression(nodeID);
			break;
		default:
			break;
	}
	return 0;

	endwin(); */
}

