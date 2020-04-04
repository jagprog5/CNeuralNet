#include "demos.h"

int main(int argc, char **argv) {
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
}