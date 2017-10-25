#ifndef FBLUR_H
#define FBLUR_H
#include "Image.h"
#include <time.h>
// #include "math.h"

// const int filterl= 5;
// const int filterR = 2;

static inline int iDivUp(int a, int b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

struct timespec remapp_time_start, remapp_time_finish;
struct timespec blur_time_start, blur_time_finish;
struct timespec upsample_time_start, upsample_time_finish;
struct timespec downsample_time_start, downsample_time_finish;
struct timespec subtract_time_start, subtract_time_finish;

double remapp_time_elapsed;
double blur_time_elapsed;
double upsample_time_elapsed;
double downsample_time_elapsed;
double subtract_time_elapsed;



void PyramidUp_sub_GPU(int imageW, int imageH, pixel_t * img, pixel_t * h_Buffr,pixel_t * target,pixel_t * subtracted,int c_off, int r_off,pixel_t * h_Buff);
void PyramidDown_GPU(int imageW, int imageH, pixel_t * img, pixel_t * h_Buffr,pixel_t * target,int c_off, int r_off,pixel_t * h_Buff);
void PyramidUp_add_GPU(int imageW, int imageH, pixel_t * img, pixel_t * h_Buffr,pixel_t * target,pixel_t * subtracted,int c_off, int r_off,pixel_t * h_Buff);

void PyramidUp_sub(int imageW, int imageH, pixel_t * img, pixel_t * h_Buffr,pixel_t * target,pixel_t * subtracted,int c_off, int r_off, int targetW, int targetH);
void PyramidDown(int imageW, int imageH, pixel_t * img, pixel_t * h_Buffr,pixel_t * target,int c_off, int r_off);
void PyramidUp_add(int imageW, int imageH, pixel_t * img, pixel_t * h_Buffr,pixel_t * target,pixel_t * subtracted,int c_off, int r_off);


#endif
