#ifndef LOCAL_LAPLACIAN_H
#define LOCAL_LAPLACIAN_H
#include "Image.h"
//#include "ap_cint.h"
/*Image read_pgm(const char * path);
void write_pgm(Image img, const char * path);
void free_pgm(Image img);*/
double alpha_, beta_;

typedef double op_t;

typedef struct{
    int start;
    int end;
} range;
Image im;
pixel_t *pim;
pixel_t *c_Buff;
range row_range,col_range;
pixel_t * paris_llf(int img_w,int img_h,pixel_t *global_pointer, double alpha, double beta,double sigma);
//Image Subsample(Image im, int boxWidth, int boxHeight, int offsetX, int offsetY) {

#endif


