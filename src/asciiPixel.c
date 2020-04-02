char shade(float pixel) {
    char c;
    if (pixel < 0.1)            c = ' ';
        else if (pixel < 0.2)   c = '.';
        else if (pixel < 0.3)   c = ':';
        else if (pixel < 0.4)   c = '-';
        else if (pixel < 0.5)   c = '=';
        else if (pixel < 0.6)   c = '+';
        else if (pixel < 0.7)   c = '*';
        else if (pixel < 0.8)   c = '#';
        else if (pixel < 0.9)   c = '&';
        else                    c = '$';
    return c;
}