
#include <stdio.h>  
#include <math.h>
#include "conv2D.cl"
#include "upsample.cl"
#include "subsample.cl"

#include "image_add_sub.cl"


#define convKernelRadius_local 2
#define convRowTileWidth_local 128
#define convKernelRadiusAligned_local 16
#define convColumnTileWidth_local 16
#define convColumnTileHeight_local 48
size_t szGlobalWorkSize1_Conv2D_local[2], szGlobalWorkSize2_Conv2D_local[2];
size_t szLocalWorkSize1_Conv2D_local[2], szLocalWorkSize2_Conv2D_local[2];

size_t szGlobalWorkSize_subSample[2];
size_t szLocalWorkSize_subSample[2];

size_t szGlobalWorkSize_upSample[2];
size_t szLocalWorkSize_upSample[2];

size_t szGlobalWorkSize_Sub[2];
size_t szLocalWorkSize_Sub[2];

void SetSubsampleSize(int targetW,int targetH ,int Gaussianwidth){
	szGlobalWorkSize_subSample[0] = targetW*Gaussianwidth;
  	szGlobalWorkSize_subSample[1] = 1;
  	szLocalWorkSize_subSample[0]  = targetW;
  	szLocalWorkSize_subSample[1]  = targetH;

}
void SetupsampleSize(int targetW,int targetH ,int Gaussianwidth){
	szGlobalWorkSize_upSample[0] = targetW*Gaussianwidth;
  	szGlobalWorkSize_upSample[1] = 1;
  	szLocalWorkSize_upSample[0]  = targetW;
  	szLocalWorkSize_upSample[1]  = targetH;

}
void SetSubSize(int targetW,int targetH ,int Gaussianwidth){
	szGlobalWorkSize_Sub[0] = targetH*targetW*Gaussianwidth;
  	szLocalWorkSize_Sub[0]  = targetW;
  	

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SetConvolutionSize_local(int width, int height) {

  szGlobalWorkSize1_Conv2D_local[0] = (convKernelRadiusAligned_local + convRowTileWidth_local + convKernelRadius_local) * iDivUp(width, convRowTileWidth_local);
  szGlobalWorkSize1_Conv2D_local[1] = 1 * height;
  szLocalWorkSize1_Conv2D_local[0]  = convKernelRadiusAligned_local + convRowTileWidth_local + convKernelRadius_local;
  szLocalWorkSize1_Conv2D_local[1]  = 1;

  szGlobalWorkSize2_Conv2D_local[0] = convColumnTileWidth_local * iDivUp(width, convColumnTileWidth_local);
  szGlobalWorkSize2_Conv2D_local[1] = 8 * iDivUp(height, convColumnTileHeight_local);
  szLocalWorkSize2_Conv2D_local[0]  = convColumnTileWidth_local;
  szLocalWorkSize2_Conv2D_local[1]  = 8;
}

void local_ConvBlur_GPU(int imageW, int imageH, double * img,double * target,double * h_Buff,int num_of_images){
	SetConvolutionSize_local(imageW,imageH);
	//double *h_Buff = (double *)malloc(imageH*imageW * sizeof(double));


	char profInfo[50];
	//sprintf(profInfo," % d convolutionRow",imageH);

	#pragma acl task inout(h_Buff) inout(img) workers(szLocalWorkSize1_Conv2D_local[0],szLocalWorkSize1_Conv2D_local[1]) groups(szGlobalWorkSize1_Conv2D_local[0],szGlobalWorkSize1_Conv2D_local[1]) label("convolutionRowGPU_local") taskid(profInfo) bind(1)  
	convolutionRowGPUlocal(h_Buff,img,imageW,imageH,num_of_images);
	#pragma acl taskwait label("convolutionRowGPU_local")

	int  smemStride = convColumnTileWidth_local * szLocalWorkSize2_Conv2D_local[1];
	int  gmemStride = imageW * szLocalWorkSize2_Conv2D_local[1];

	#pragma acl task inout(h_Buff) out(target) workers(szLocalWorkSize2_Conv2D_local[0],szLocalWorkSize2_Conv2D_local[1]) groups(szGlobalWorkSize2_Conv2D_local[0],szGlobalWorkSize2_Conv2D_local[1]) label("convolutionColumnGPU_local") taskid(profInfo) bind(1) 
	convolutionColumnGPUlocal(target,h_Buff,imageW,imageH,smemStride,gmemStride,num_of_images);
	#pragma acl taskwait label("convolutionColumnGPU_local")

}

void local_Upsample(double *d_Result,double *d_Input,int y_offset,int ResultW,int ResultH,int l,int hw,int gaussianW ){
	char profInfo[50];
	SetupsampleSize(ResultW/2+ResultW%2,ResultH/2+ResultH%2 ,gaussianW);
	// #pragma acl task inout(d_Result) inout(d_Input) workers(szLocalWorkSize_upSample[0],szLocalWorkSize_upSample[1]) groups(szGlobalWorkSize_upSample[0],szGlobalWorkSize_upSample[1])label("Upsample_GPU") taskid(profInfo) bind(1)
	// UpsampleGPU( d_Result,   d_Input,  y_offset,  ResultW, ResultH,  l,  hw);
	// #pragma acl taskwait label("Upsample_GPU")
}

void local_Downsample( double * target , double * temp,int imageH,int  imageW,int  y_offset,int  l ,int hw,int  gaussianW){
	char profInfo[50];
	SetSubsampleSize(imageW/2+imageW%2,imageH/2+imageH%2 ,gaussianW);

	#pragma acl task inout(target) inout(temp) workers(szLocalWorkSize_subSample[0],szLocalWorkSize_subSample[1]) groups(szGlobalWorkSize_subSample[0],szGlobalWorkSize_subSample[1]) label("downsampleGPU") taskid(profInfo) bind(1)  
	downsampleGPU(target,temp,     y_offset,      imageW,   imageH,     l,     hw);
	#pragma acl taskwait label("downsampleGPU")
}

void local_sub(double *d_Result,double *d_Input,int targetW,int targetH,gaussianW){
	char profInfo[50];
	SetSubSize(targetW,targetH,gaussianW);
	
	#pragma acl task inout(d_Result) inout(d_Input) workers(szLocalWorkSize_Sub[0]) groups(szGlobalWorkSize_Sub[0])label("Sub_GPU") taskid(profInfo) bind(1)
	Sub_GPU(d_Result,d_Input);
	#pragma acl taskwait label("Sub_GPU")
}
void local_PyramidDown(int imageW, int imageH, double * img, double * h_Buff,double * target,double * temp,int y_offset, int l, int hw,int gaussianW){
	local_ConvBlur_GPU( imageW,  imageH, img, temp, h_Buff, gaussianW);
	printf("blur\n");
	local_Downsample( target ,  temp, imageH,  imageW,y_offset,l,hw,gaussianW);
	printf("downsample\n");
}



void local_PyramidUp_sub( double * img, double * temp,double * target,double * subtracted,double * h_Buff,int y_offset, int l, int hw,int gaussianW, int targetW, int targetH){

	local_Upsample(temp ,  img, y_offset, targetW, targetH,l,hw,gaussianW);
	local_ConvBlur_GPU( targetW,  targetH, temp, target, h_Buff, gaussianW);
	local_sub( target ,  subtracted  , targetH, targetW,gaussianW);
}
