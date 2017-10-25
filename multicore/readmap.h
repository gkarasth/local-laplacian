#ifndef READMAP_H
#define READMAP_H
#include "Image.h"
typedef struct {
    
    int isLoaded;
    int bpp;
    int width, height;
    unsigned char* pixels;

    
} FULLImage;


typedef struct {
	double R;
	double G;
	double B;
} mapper;

#define EPSILON 1.192093e-07
void FULLImageF(FULLImage* inst,char* fileName,Image *inputimg);
Image resize_image(Image *inputimg);
void FULLImagewrite(FULLImage* img,Image inputimg,char *out_filename);
#endif