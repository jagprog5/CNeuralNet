#ifndef ASCII_PIXEL_H
#define ASCII_PIXEL_H

#define DEF_PAIR 1
#define GOOD_PAIR 2
#define BAD_PAIR 3

int y_cursor;
int x_cursor;

#define set_cursor() move(y_cursor, x_cursor)

void init_color_pairs();

char shade(float pixel);

#endif
