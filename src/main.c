#include <stdlib.h>
#include <stdio.h>
#include "curses.h"

#define DEFAULT_PAIR 1
#define COLOR_BG COLOR_MAGENTA

int main(int argc, char **argv) {
	if (initscr() == NULL) {
		fprintf(stderr, "Error initialising ncurses.\n");
		exit(1);
	}
	start_color();
	init_color_pairs();
	init_color(COLOR_BG, 0, 0, 0);//20, 20, 100);
	init_pair(DEFAULT_PAIR, COLOR_WHITE, COLOR_BG);
	cbreak();
	noecho();
	keypad(stdscr, TRUE);



	//nodelay(stdscr, TRUE);
	//start_color();
	border(0, 0, 0, 0, 0, 0, 0, 0);
	move(5,5);
	addstr("tester"); 
	if (has_colors()) {
		move(6,5);
		addstr("Can change color");
	}
	refresh();
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

