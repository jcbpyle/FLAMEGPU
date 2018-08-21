#ifndef __VISUALISATION_H
#define __VISUALISATION_H

//#define PAUSE_ON_START

// constants
const unsigned int WINDOW_WIDTH = 512;
const unsigned int WINDOW_HEIGHT = 512;

//frustrum
const float NEAR_CLIP = 0.1f;
const float FAR_CLIP = 100.0f;

//Circle model fidelity
const int SPHERE_SLICES = 20;
const int SPHERE_STACKS = 20;
const float SPHERE_RADIUS = 0.0025f;

//Viewing Distance
const float VIEW_DISTANCE = 1.5f;

//light position
GLfloat LIGHT_POSITION[] = {10.0f, 10.0f, 10.0f, 1.0f};

#endif //__VISUALISATION_H
