#ifndef FBLUR_H
#define FBLUR_H
#include "Image.h"
// #include "math.h"

// const int filterl= 5;
// const int filterR = 2;

static inline int iDivUp(int a, int b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}





void PyramidUp_sub_GPU(int imageW, int imageH, pixel_t * img, pixel_t * h_Buffr,pixel_t * target,pixel_t * subtracted,int c_off, int r_off,pixel_t * h_Buff);
void PyramidDown_GPU(int imageW, int imageH, pixel_t * img, pixel_t * h_Buffr,pixel_t * target,int c_off, int r_off,pixel_t * h_Buff);
void PyramidUp_add_GPU(int imageW, int imageH, pixel_t * img, pixel_t * h_Buffr,pixel_t * target,pixel_t * subtracted,int c_off, int r_off,pixel_t * h_Buff);

void PyramidUp_sub(int imageW, int imageH, pixel_t * img, pixel_t * h_Buffr,pixel_t * target,pixel_t * subtracted,int c_off, int r_off, int targetW, int targetH);
void PyramidDown(int imageW, int imageH, pixel_t * img, pixel_t * h_Buffr,pixel_t * target,int c_off, int r_off);
void PyramidUp_add(int imageW, int imageH, pixel_t * img, pixel_t * h_Buffr,pixel_t * target,pixel_t * subtracted,int c_off, int r_off);


#endif
