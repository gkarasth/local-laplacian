#include "fast_blur.h"
#include <stdio.h>  
#include <math.h>
#include "conv2D.cl"

#define convKernelRadius 2
#define convRowTileWidth 128
#define convKernelRadiusAligned 16
#define convColumnTileWidth 16
#define convColumnTileHeight 48

size_t szGlobalWorkSize1_Conv2D[2], szGlobalWorkSize2_Conv2D[2];
size_t szLocalWorkSize1_Conv2D[2], szLocalWorkSize2_Conv2D[2];

pixel_t h_Filter[5] = {0.05, 0.25, 0.4, 0.25, 0.05};
// const double weight = 20;
// pixel_t h_Filter[5] = {1,5,8,5,1};

#define filterR 2
#define filterl (2 * filterR + 1)

// const int filterl = 5;
// const int filterR = 2;
// const int weight = 20;
// int h_Filter[5] = {1,5,8,5,1};//original [Burt and Adelson, 1983]
//original [Burt and Adelson, 1983]

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SetConvolutionSize(int width, int height) {

  szGlobalWorkSize1_Conv2D[0] = (convKernelRadiusAligned + convRowTileWidth + convKernelRadius) * iDivUp(width, convRowTileWidth);
  szGlobalWorkSize1_Conv2D[1] = 1 * height;
  szLocalWorkSize1_Conv2D[0]  = convKernelRadiusAligned + convRowTileWidth + convKernelRadius;
  szLocalWorkSize1_Conv2D[1]  = 1;

  szGlobalWorkSize2_Conv2D[0] = convColumnTileWidth * iDivUp(width, convColumnTileWidth);
  szGlobalWorkSize2_Conv2D[1] = 8 * iDivUp(height, convColumnTileHeight);
  szLocalWorkSize2_Conv2D[0]  = convColumnTileWidth;
  szLocalWorkSize2_Conv2D[1]  = 8;
}



void convolutionRow(pixel_t *h_Dst, pixel_t * h_Src, pixel_t *h_Filter, 
                       int imageW, int imageH) {

  int x, y, k;
     
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      pixel_t sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += h_Src[ y * imageW + d] * h_Filter[filterR - k];
        }     
        
      }
      h_Dst[y * imageW + x] = sum;
    }
  }
        
}
void convolutionCol(pixel_t * h_Dst, pixel_t *h_Src, pixel_t *h_Filter, 
                       int imageW, int imageH) {

  int x, y, k;

  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      pixel_t sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        }   
 
      }
      h_Dst[y * imageW + x] = sum;
    }
  }
    
}
// void ConvBlur_GPU(int imageW, int imageH, pixel_t * img,pixel_t * target,pixel_t * h_Buff){
//   SetConvolutionSize(imageW,imageH);
//   //double *h_Buff = (double *)malloc(imageH*imageW * sizeof(double));
  

//   char profInfo[50];
//   //sprintf(profInfo," % d convolutionRow",imageH);
  
//   #pragma acl task inout(h_Buff) inout(img) workers(szLocalWorkSize1_Conv2D[0],szLocalWorkSize1_Conv2D[1])groups(szGlobalWorkSize1_Conv2D[0],szGlobalWorkSize1_Conv2D[1])label("convolutionRowGPU") taskid(profInfo) bind(1)  
//   convolutionRowGPU(h_Buff,img,imageW,imageH);
//   #pragma acl taskwait label("convolutionRowGPU")
  
//   int  smemStride = convColumnTileWidth * szLocalWorkSize2_Conv2D[1];
//   int  gmemStride = imageW * szLocalWorkSize2_Conv2D[1];

//   #pragma acl task inout(h_Buff) out(target) workers(szLocalWorkSize2_Conv2D[0],szLocalWorkSize2_Conv2D[1]) groups(szGlobalWorkSize2_Conv2D[0],szGlobalWorkSize2_Conv2D[1]) label("convolutionColumnGPU") taskid(profInfo) bind(1) 
//   convolutionColumnGPU(target,h_Buff,imageW,imageH,smemStride,gmemStride);
//   #pragma acl taskwait label("convolutionColumnGPU")

// }

void ConvBlur(int imageW, int imageH, pixel_t * img,pixel_t * target){
  pixel_t *h_Buff = (pixel_t *)malloc(imageH*imageW * sizeof(pixel_t));

  convolutionRow(h_Buff, img, h_Filter, imageW, imageH); // convolution kata grammes
  convolutionCol(target, h_Buff, h_Filter, imageW, imageH); 

  free(h_Buff);
}

void add(pixel_t * target , pixel_t * add1 , pixel_t * add2 ,int imageH,int imageW){
  int x,y;
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++){
      target[y*imageW+x] = add1[y*imageW+x]+add2[y*imageW+x];
    }
  }
}

void sub(pixel_t * target , pixel_t * Sed , pixel_t * Sor ,int imageH,int imageW){
  int x,y;
  for (y = 0; y < imageH; y++) {
      for (x = 0; x < imageW; x++) {
      // target[y*(imageW+imageW%2)+x] = Sed[y*imageW+x]-Sor[y*(imageW+imageW%2)+x];
      target[y*imageW+x] = Sed[y*imageW+x]-Sor[y*imageW+x];
    }
  }
}

void Upsample(pixel_t * target , pixel_t * source,int imageH, int imageW,int c_off, int r_off,pixel_t * init){
  
  int src_imageW = imageW/2 + imageW%2;
  int src_imageH = imageH/2 + imageH%2;
  int x,y;
  for (int i = 0; i < imageW*imageH; ++i)
  {
    target[i] =0;//gp[init +i];
  }
  for (y = 0; y < src_imageH; y++) {
    for (x = 0; x < src_imageW; x++) {
      int addr0x = 2*x + c_off ;
      int addr0y = 2*y + r_off ;

      int addr1x = 2*x + 1 - c_off;  
      int addr1y = 2*y + r_off ;

      int addr2x = 2*x + c_off;
      int addr2y = 2*y + 1 - r_off; 

      int addr3x = 2*x + 1 - c_off;
      int addr3y = 2*y + 1 - r_off; 

      if(addr0y<imageH && addr0x<imageW)
        target[addr0x+addr0y*imageW]=4.0*source[x+y*src_imageW];
      if(addr1y<imageH && addr1x<imageW)
        target[addr1x+addr1y*imageW]=0;//source[x+y*src_imageW];
      if(addr2y<imageH && addr2x<imageW)
        target[addr2x+addr2y*imageW]=0;//source[x+y*src_imageW];
      if(addr3y<imageH && addr3x<imageW)
        target[addr3x+addr3y*imageW]=0;//source[x+y*src_imageW];
    }
  }
}

void Downsample( pixel_t * target ,pixel_t * source,int src_imageH, int src_imageW,int c_off, int r_off){
  int x,y;
  int dst_imH = src_imageH/2+src_imageH%2;
  int dst_imW = src_imageW/2+src_imageW%2;
  
  for (y = 0; y < dst_imH; y++) {
    for (x = 0; x < dst_imW; x++) {
      if(2*x + c_off>=src_imageW || 2*y + r_off>=src_imageH){
        target[y*dst_imW +x ] = 0;
      }
      else{
        target[y*dst_imW +x ] = source[ y*2*src_imageW+x*2+(r_off*src_imageW)+c_off]; 
      }
    }
  }
}



void PyramidDown(int imageW, int imageH, pixel_t * img, pixel_t * h_Buffr,pixel_t * target,int c_off, int r_off){
  ConvBlur( imageW,  imageH,  img, h_Buffr);
  
  Downsample( target ,  h_Buffr, imageH,  imageW,c_off,r_off);
  
}


void PyramidUp_sub(int imageW, int imageH, pixel_t * img, pixel_t * h_Buffr,pixel_t * target,pixel_t * subtracted,int c_off, int r_off, int targetW, int targetH){
  Upsample(h_Buffr ,  img,  targetH, targetW,c_off,r_off,subtracted);
  
  ConvBlur( targetW,  targetH,  h_Buffr, target);
  
  sub( target ,  subtracted ,  target , targetH, targetW);
  
}
// void PyramidUp_add(int imageW, int imageH, pixel_t * img, pixel_t * h_Buffr,pixel_t * target,pixel_t * subtracted,int c_off, int r_off){

//  Upsample(h_Buffr ,  img, imageH*2,  imageW*2,c_off,r_off,subtracted);
//  ConvBlur( imageW*2,  imageH*2,  h_Buffr, h_Buffr);
//  add( target ,  subtracted ,  h_Buffr , imageH*2, imageW*2);
// }
void PyramidUp_add(int imageW, int imageH, pixel_t * img, pixel_t * h_Buffr,pixel_t * target,pixel_t * subtracted,int c_off, int r_off){

  Upsample(h_Buffr ,  img, imageH,  imageW,c_off,r_off,subtracted);
  ConvBlur( imageW,  imageH,  h_Buffr, h_Buffr);
  add( target ,  subtracted ,  h_Buffr , imageH, imageW);
}

// void PyramidDown_GPU(int imageW, int imageH, pixel_t * img, pixel_t * temp,pixel_t * target,int c_off, int r_off,pixel_t * h_Buff){
//   ConvBlur_GPU( imageW,  imageH,  img, temp,h_Buff);
//   Downsample( target ,  temp, imageH/2,  imageW/2,c_off,r_off);
// }

// void PyramidUp_sub_GPU(int imageW, int imageH, pixel_t * img, pixel_t * temp,pixel_t * target,pixel_t * subtracted,int c_off, int r_off,pixel_t * h_Buff){

//   Upsample(temp ,  img, imageH,  imageW,c_off,r_off,subtracted);
//   ConvBlur_GPU( imageW*2,  imageH*2,  temp, target,h_Buff);
//   sub( target ,  subtracted ,  target , imageH*2, imageW*2);
// }
// void PyramidUp_add_GPU(int imageW, int imageH, pixel_t * img, pixel_t * temp,pixel_t * target,pixel_t * subtracted,int c_off, int r_off,pixel_t * h_Buff){

//   Upsample(temp ,  img, imageH,  imageW,c_off,r_off,subtracted);
//   ConvBlur_GPU( imageW*2,  imageH*2,  temp, temp,h_Buff);
//   add( target ,  subtracted ,  temp , imageH*2, imageW*2);
// }