#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>                                                                                                                                                                                                                                                                                                       
#include "paris_llf.h"
#include "Image.h"
#include "fast_blur.h"
#include "remapping_function.cl"
#include "remap.c"
#include <time.h>
//#include "gpu_pyramid.c"
#include "conv2D.cl"
#include "upsample.cl"
#include "subsample.cl"
#include "image_add_sub.cl"

#define DEVICE_REMAPP 1
#define DEVICE_CONVROW 1
#define DEVICE_CONVCOL 1
#define DEVICE_UPSAMPLE 1
#define DEVICE_DOWNSAMPLE 1
#define DEVICE_SUB 1

#define convKernelRadius_local 2
#define convRowTileWidth_local 128
#define convKernelRadiusAligned_local 16
#define convColumnTileWidth_local 16
#define convColumnTileHeight_local 48


static inline int intDivUp(int a, int b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

size_t szGlobalWorkSize1_Conv2D_local[2], szGlobalWorkSize2_Conv2D_local[2];
size_t szLocalWorkSize1_Conv2D_local[2], szLocalWorkSize2_Conv2D_local[2];

size_t szGlobalWorkSize_subSample[2];
size_t szLocalWorkSize_subSample[2];

size_t szGlobalWorkSize_upSample[2];
size_t szLocalWorkSize_upSample[2];

size_t szGlobalWorkSize_Sub[2];
size_t szLocalWorkSize_Sub[2];

size_t szGlobalWorkSize_Remap[2];
size_t szLocalWorkSize_Remap[2];

struct timespec remapp_time_start, remapp_time_finish;
struct timespec blur_time_start, blur_time_finish;
struct timespec upsample_time_start, upsample_time_finish;
struct timespec downsample_time_start, downsample_time_finish;
struct timespec subtract_time_start, subtract_time_finish;


double time_elapsed;
double remapp_time_elapsed;
double blur_time_elapsed;
double upsample_time_elapsed;
double downsample_time_elapsed;
double subtract_time_elapsed;

void SetRemappingSize(int Gaussianwidth, int hw,int l) {
int tile_width = 132;
while(Gaussianwidth%(tile_width - 2*hw)/(1<<l)!=0||(tile_width - 2*hw)%(1<<l)){
	tile_width+=1;
}
printf("%d\n",tile_width);
  szGlobalWorkSize_Remap[0] = tile_width*intDivUp(Gaussianwidth,(tile_width - 2*hw)/(1<<l));
  szGlobalWorkSize_Remap[1] = 1;
  szLocalWorkSize_Remap[0]  = tile_width;
  szLocalWorkSize_Remap[1]  = 1;
}

void SetSubsampleSize(int targetW,int targetH ,int Gaussianwidth){
	szGlobalWorkSize_subSample[0] = targetW*Gaussianwidth;
  	szGlobalWorkSize_subSample[1] = targetH;
  	szLocalWorkSize_subSample[0]  = targetW;
  	szLocalWorkSize_subSample[1]  = 1;

}
void SetupsampleSize(int srcW,int srcH ,int Gaussianwidth){
	szGlobalWorkSize_upSample[0] = srcW*Gaussianwidth;
  	szGlobalWorkSize_upSample[1] = srcH;
  	szLocalWorkSize_upSample[0]  = srcW;
  	szLocalWorkSize_upSample[1]  = 1;

}
void SetSubSize(int targetW,int targetH ,int Gaussianwidth){
	szGlobalWorkSize_Sub[0] = targetH*targetW*Gaussianwidth;
  	szLocalWorkSize_Sub[0]  = targetW;
  	

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// void SetConvolutionSize_local(int width, int height) {

//   szGlobalWorkSize1_Conv2D_local[0] = (convKernelRadiusAligned_local + convRowTileWidth_local + convKernelRadius_local) * iDivUp(width, convRowTileWidth_local);
//   szGlobalWorkSize1_Conv2D_local[1] = 1 * height;
//   szLocalWorkSize1_Conv2D_local[0]  = convKernelRadiusAligned_local + convRowTileWidth_local + convKernelRadius_local;
//   szLocalWorkSize1_Conv2D_local[1]  = 1;

//   szGlobalWorkSize2_Conv2D_local[0] = convColumnTileWidth_local * iDivUp(width, convColumnTileWidth_local);
//   szGlobalWorkSize2_Conv2D_local[1] = 8 * iDivUp(height, convColumnTileHeight_local);
//   szLocalWorkSize2_Conv2D_local[0]  = convColumnTileWidth_local;
//   szLocalWorkSize2_Conv2D_local[1]  = 8;
// }

void SetConvolutionSize_local(int width, int height , int Gaussianwidth) {

  szGlobalWorkSize1_Conv2D_local[0] = (width+2*2)*Gaussianwidth;
  szGlobalWorkSize1_Conv2D_local[1] = height;
  szLocalWorkSize1_Conv2D_local[0]  = (width+2*2);
  szLocalWorkSize1_Conv2D_local[1]  = 1;


  szGlobalWorkSize2_Conv2D_local[0] = (height+2*2)*Gaussianwidth;
  szGlobalWorkSize2_Conv2D_local[1] = width;
  szLocalWorkSize2_Conv2D_local[0]  = (height+2*2);
  szLocalWorkSize2_Conv2D_local[1]  = 1;
  //printf("width %d , height %d\n",width,height);
}

void local_ConvBlur_GPU(int imageW, int imageH, pixel_t * img,pixel_t * target,pixel_t * h_Buff,int num_of_images,int hw ,int l,int j,int w){
	SetConvolutionSize_local(imageW,imageH,num_of_images);
	//double *h_Buff = (double *)malloc(imageH*imageW * sizeof(double));


	char profInfo[50];
	//sprintf(profInfo," % d convolutionRow",imageH);
	#pragma acl task in(h_Buff) in(img) workers(szLocalWorkSize1_Conv2D_local[0],szLocalWorkSize1_Conv2D_local[1]) groups(szGlobalWorkSize1_Conv2D_local[0],szGlobalWorkSize1_Conv2D_local[1]) label("convolutionRowGPU_local") taskid(profInfo) bind(DEVICE_CONVROW)  
	convolutionRowGPUlocal(h_Buff,img,imageW,imageH,hw,l,j,im.w,num_of_images);
	#pragma acl taskwait label("convolutionRowGPU_local")



	#pragma acl task in(h_Buff) inout(target) workers(szLocalWorkSize2_Conv2D_local[0],szLocalWorkSize2_Conv2D_local[1]) groups(szGlobalWorkSize2_Conv2D_local[0],szGlobalWorkSize2_Conv2D_local[1]) label("convolutionColumnGPU_local") taskid(profInfo) bind(DEVICE_CONVCOL) 
	convolutionColumnGPUlocal(target,h_Buff,imageW,imageH,hw,l,j,im.w,num_of_images);
	#pragma acl taskwait label("convolutionColumnGPU_local")

}

void local_Upsample(pixel_t *d_Result,pixel_t *d_Input,int y_offset,int ResultW,int ResultH,int l,int j,int hw,int gaussianW ){
	char profInfo[50];
	SetupsampleSize(ResultW/2+ResultW%2,ResultH/2+ResultH%2 ,gaussianW);

	#pragma acl task inout(d_Result) in(d_Input) workers(szLocalWorkSize_upSample[0],szLocalWorkSize_upSample[1]) groups(szGlobalWorkSize_upSample[0],szGlobalWorkSize_upSample[1])label("Upsample_GPU") taskid(profInfo) bind(DEVICE_UPSAMPLE)
	UpsampleGPU( d_Result,   d_Input,  y_offset,  ResultW, ResultH,im.w,  l, j, hw);
	#pragma acl taskwait label("Upsample_GPU")
}

void local_Downsample( pixel_t * target , pixel_t * temp,int imageH,int  imageW,int  y_offset,int  l ,int j,int hw,int  gaussianW){
	char profInfo[50];
	SetSubsampleSize(imageW/2+imageW%2,imageH/2+imageH%2 ,gaussianW);

	#pragma acl task inout(target) in(temp) workers(szLocalWorkSize_subSample[0],szLocalWorkSize_subSample[1]) groups(szGlobalWorkSize_subSample[0],szGlobalWorkSize_subSample[1]) label("downsampleGPU") taskid(profInfo) bind(DEVICE_DOWNSAMPLE)  
	downsampleGPU(target,temp,     y_offset,      imageW,   imageH,im.w,     l, j,    hw);
	#pragma acl taskwait label("downsampleGPU")
}

void local_sub(pixel_t *d_Result,pixel_t *d_Input,int targetW,int targetH,int gaussianW){
	char profInfo[50];
	SetSubSize(targetW,targetH,gaussianW);
	
	#pragma acl task device_out(d_Result) in(d_Input) workers(szLocalWorkSize_Sub[0]) groups(szGlobalWorkSize_Sub[0])label("Sub_GPU") taskid(profInfo) bind(DEVICE_SUB)
	Sub_GPU(d_Result,d_Input);
	#pragma acl taskwait label("Sub_GPU")
}
void local_PyramidDown(int imageW, int imageH, pixel_t * img, pixel_t * temp,pixel_t * target,pixel_t * h_Buff,int y_offset, int l,int j, int hw,int gaussianW){



	clock_gettime(CLOCK_MONOTONIC, &blur_time_start);
	
	local_ConvBlur_GPU( imageW,  imageH, img, temp, h_Buff, gaussianW,hw , l,j ,-1);


	clock_gettime(CLOCK_MONOTONIC, &blur_time_finish);
  	blur_time_elapsed += (blur_time_finish.tv_sec - blur_time_start.tv_sec);
  	blur_time_elapsed += (blur_time_finish.tv_nsec - blur_time_start.tv_nsec) / 1000000000.0;



  	clock_gettime(CLOCK_MONOTONIC, &downsample_time_start);	

	local_Downsample( target ,  temp, imageH,  imageW,y_offset,l,j,hw,gaussianW);
	
	clock_gettime(CLOCK_MONOTONIC, &downsample_time_finish);
  	downsample_time_elapsed += (downsample_time_finish.tv_sec - downsample_time_start.tv_sec);
  	downsample_time_elapsed += (downsample_time_finish.tv_nsec - downsample_time_start.tv_nsec) / 1000000000.0;


	//local_Downsample( target ,  img, imageH,  imageW,y_offset,l,j,hw,gaussianW);
}



void local_PyramidUp_sub( int targetW, int targetH, pixel_t * img, pixel_t * temp,pixel_t * target,pixel_t * subtracted,pixel_t * h_Buff,int y_offset, int l,int j, int hw,int gaussianW){
	clock_gettime(CLOCK_MONOTONIC, &upsample_time_start);

	local_Upsample(temp ,  img, y_offset, targetW, targetH,l,j,hw,gaussianW);

  	clock_gettime(CLOCK_MONOTONIC, &upsample_time_finish);
  	upsample_time_elapsed += (upsample_time_finish.tv_sec - upsample_time_start.tv_sec);
  	upsample_time_elapsed += (upsample_time_finish.tv_nsec - upsample_time_start.tv_nsec) / 1000000000.0;

 

	clock_gettime(CLOCK_MONOTONIC, &blur_time_start);
	
	local_ConvBlur_GPU( targetW,  targetH, temp, target, h_Buff, gaussianW,hw ,l,j,-1);
  	
  	clock_gettime(CLOCK_MONOTONIC, &blur_time_finish);
  	blur_time_elapsed += (blur_time_finish.tv_sec - blur_time_start.tv_sec);
  	blur_time_elapsed += (blur_time_finish.tv_nsec - blur_time_start.tv_nsec) / 1000000000.0;


	// clock_gettime(CLOCK_MONOTONIC, &subtract_time_start);

	// local_sub( target ,  subtracted  , targetH, targetW,gaussianW);

	// clock_gettime(CLOCK_MONOTONIC, &subtract_time_finish);
 //  	subtract_time_elapsed += (subtract_time_finish.tv_sec - subtract_time_start.tv_sec);
 //  	subtract_time_elapsed += (subtract_time_finish.tv_nsec - subtract_time_start.tv_nsec) / 1000000000.0;
	// local_Upsample(target ,  img, y_offset, targetW, targetH,l,j,hw,gaussianW);
	// local_sub( target ,  subtracted  , targetH, targetW,gaussianW);
}



const int  J = 5;
#ifndef max
  #define max(a, b) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
  #define min(a, b) ( ((a) < (b)) ? (a) : (b) )
#endif

double SmoothStep(double x_min, double x_max, double x) {
  double y = (x - x_min) / (x_max - x_min);
  y = max(0.0, min(1.0, y));
  return pow(y, 2) * pow(y-2, 2);
}

double DetailRemap(double delta, double sigma_r) {
  double fraction = delta / sigma_r;
  double polynomial = pow(fraction, alpha_);
  if (alpha_ < 1) {
    const double kNoiseLevel = 0.01;
    double blend = SmoothStep(kNoiseLevel,2 * kNoiseLevel, fraction * sigma_r);
    polynomial = blend * polynomial + (1 - blend) * fraction;
  }
  return polynomial;
}

double EdgeRemap(double delta) {
  return beta_ * delta;
}
void RemappingFunction(double value,
                                 double reference,
                                 double sigma_r,
                                 int output_dst,
                                 pixel_t * dst) {
  	double delta = reference - value;
  	if(value>reference)
  		delta = value - reference;
  	
  	int sign = value < reference ? -1 : 1;
  	
	double temp;
  	if (delta < sigma_r) {
    	temp = reference + sign * sigma_r * DetailRemap(delta, sigma_r);     	

  	} else {
    	temp = reference + sign * (EdgeRemap(delta - sigma_r) + sigma_r);	
  	}
  	dst[output_dst] =(pixel_t) temp;
  
}
void remapp(range row_range ,range col_range ,int col_size, int row_size, pixel_t  * dst ,pixel_t reference ,double sigma){
	int x,y;

	pixel_t input;
	for (y = 0; y < row_size ; ++y)
	{
		for (x = 0; x < col_size; ++x)
		{	

			int output_dst = x+y*col_size;
			if (y+row_range.start>=im.h||x+col_range.start>=im.w||x+col_range.start<0||y+row_range.start<0)
			{
				
				dst[output_dst]=0;
			}
			else
			{
				input = pim[ col_range.start +x +(y+row_range.start)*im.w];
				RemappingFunction((double)input,(double)reference,sigma,output_dst,dst);

			}
	
			
			
			
		}
	}
}

pixel_t *  paris_llf(int img_w,int img_h,pixel_t *global_pointer, double alpha, double beta,double sigma){
  	pixel_t * pPyramid[J];
	pixel_t * pLPyramid[J];

	//int * global_pointer;
    alpha_=alpha;
    beta_ =beta ;
    im.w = img_w;
    im.h = img_h;
    im.img = 0;
    pim = global_pointer;
    
    int j,i,k;
	/////////////////////////////////////////////////////////////////////address manager ///////////////////////////////////
    int Level_w[J];
    int Level_h[J];
    int Level_size[J];
    // compute the size of the image at every pyramid level 
    int pyramid_size ;
    Level_w[0] = im.w;
    Level_h[0] = im.h;
    Level_size[0] = im.w*im.h;
    
    time_elapsed=0;
    remapp_time_elapsed=0;
	blur_time_elapsed=0;
	upsample_time_elapsed=0;
	downsample_time_elapsed=0;
	subtract_time_elapsed=0;
    


	printf("%d %d \n", Level_w[0],Level_h[0]);
    pyramid_size = Level_size[0];
    for (i = 1; i < J; ++i)
    {
        Level_w[i] = Level_w[i-1]/2+Level_w[i-1]%2;
        Level_h[i] = Level_h[i-1]/2+Level_h[i-1]%2;
        printf("%d %d \n", Level_w[i],Level_h[i]);
        Level_size[i] = Level_w[i]*Level_h[i];
        pyramid_size += Level_size[i];
    }
	// compute addresses for input pyrpixel_tamids//////////////////////////////////////////////////////////////////////////////////////////////////
	
    

    //Gaussian/////////////////////
	    Image imPyramid[J];
	   


	    imPyramid[0].img = pim;
	    imPyramid[0].w = Level_w[0];
	    imPyramid[0].h = Level_h[0];
	    
	    for (i = 1; i < J; ++i)
	    {
	       
	       imPyramid[i].img = (pixel_t *) malloc(Level_size[i]*sizeof(pixel_t));
	       imPyramid[i].w = Level_w[i];
	       imPyramid[i].h = Level_h[i];
	    }
	///////////////////////////////
printf("malloc for Gaussian done\n");

    //temp for Gaussian-Laplacian////////////
	    Image imTempPyramid[J];
	    
	    
	

	    imTempPyramid[0].img = (pixel_t *) malloc(Level_size[0]*sizeof(pixel_t));
	    imTempPyramid[0].h = Level_w[0];
	    imTempPyramid[0].w = Level_h[0];
	
		
		pLPyramid[0] = (pixel_t *)malloc(Level_size[0]/4*sizeof(pixel_t));
		pPyramid[0] = (pixel_t *)malloc(Level_size[0]/4*sizeof(pixel_t)); 
	    for (i = 1; i < J; ++i)
	    {
	    	pLPyramid[i] = (pixel_t *)malloc(Level_size[i]/4*sizeof(pixel_t));
	   		pPyramid[i] = (pixel_t *)malloc(Level_size[i]/4*sizeof(pixel_t)); 
			
	    	imTempPyramid[i].img = (pixel_t *) malloc(Level_size[i]*sizeof(pixel_t));
	    	imTempPyramid[i].h = Level_w[i];
	    	imTempPyramid[i].w = Level_h[i];
	    }
	///////////////////////////////
printf("malloc for TempPyramid done\n");


    //Laplacian////////////////////
	    Image imLPyramid[J];




	    imLPyramid[0].img = (pixel_t *) malloc(Level_size[0]*sizeof(pixel_t));;
	    imLPyramid[0].w = Level_w[0];
	    imLPyramid[0].h = Level_h[0];
   	      

	    for (i = 1; i < J; ++i)
	    {
	       	imLPyramid[i].img = (pixel_t *) malloc(Level_size[i]*sizeof(pixel_t));
   	    	
	       	imLPyramid[i].w = Level_w[i];
	       	imLPyramid[i].h = Level_h[i];
	    }
    //////////////////////////////
printf("malloc for imLPyramid done\n");


    	Image Pyramid[J];
	    
	    

   

	////////////////////////////////////////////////////////////////////

  	const int kRows = im.h;
  	const int kCols = im.w;
	int cstart =0;
	int rstart =0;
    // Compute a Gaussian and Laplacian pyramid for the input



#if 0   
	double * buffer = (double *) malloc(Level_size[0]*sizeof(double));
    for (j = 1; j < J; j++) {
        PyramidDown_GPU(imPyramid[j-1].w, imPyramid[j-1].h, imPyramid[j-1].img, imTempPyramid[j-1].img , imPyramid[j].img,0,0,buffer);
	}
#endif
#if 1
    for (j = 1; j < J; j++) {
        PyramidDown(imPyramid[j-1].w, imPyramid[j-1].h, imPyramid[j-1].img, imTempPyramid[j-1].img , imPyramid[j].img,0,0);
	}
#endif
    for (i = 0; i < imLPyramid[J-1].w*imLPyramid[J-1].h; ++i)
    {
    	imLPyramid[J-1].img[i] = imPyramid[J-1].img[i];
    }
 
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	
	int subimage_max_size = (3*(1<<J)-2)*2;
	// int address_space_size = subimage_max_size*subimage_max_size*imLPyramid[J-1].w;
	// double * pyramid_address_space = (double *) malloc(address_space_size*sizeof(double));
	
	int address_space_size[J];
	pixel_t * 	pyramid_address_space[J];
	pixel_t *	G_pyramid_address_space[J];
	pixel_t *	L_pyramid_address_space[J];
	pixel_t *	Temp_pyramid_address_space[J];
	pixel_t *	Buff_pyramid_address_space[J];
	for (int i = 0; i < J; ++i)
	{
		address_space_size[i] = subimage_max_size*subimage_max_size*imLPyramid[J-1].w;
		G_pyramid_address_space[i] 		= 	(pixel_t *) malloc(address_space_size[i]*sizeof(pixel_t));
		L_pyramid_address_space[i] 		= 	(pixel_t *) malloc(address_space_size[i]*sizeof(pixel_t));
		Temp_pyramid_address_space[i] 	= 	(pixel_t *) malloc(address_space_size[i]*sizeof(pixel_t));
		Buff_pyramid_address_space[i] 	= 	(pixel_t *) malloc(address_space_size[i]*sizeof(pixel_t));
		subimage_max_size = subimage_max_size/2 + subimage_max_size%2;
	}


	// if(pyramid_address_space == NULL)
	// {
	// 	printf("Memory allocation failed");
	// 	return(0);
	// }
	

	Image LPyramid[J];
    
   
	struct timespec time_start, time_finish;
	double time_elapsed_t[J];
	double remapp_time_elapsed_t[J];
	double blur_time_elapsed_t[J];
	double upsample_time_elapsed_t[J];
	double downsample_time_elapsed_t[J];
	double subtract_time_elapsed_t[J];
	clock_gettime(CLOCK_MONOTONIC, &time_start); 
    
    
    for (int l = 0; l < J-1; l++) {
		int hw = 3*(1<<(l+1))-2;

    	printf("level = %d ,hw =%d  ,last %d element %d\n",l ,hw, hw/(1<<(l+1)), hw>>l);
    	SetRemappingSize( imLPyramid[l].w, hw, l ) ;
    	printf("%d %d \n",(int)szLocalWorkSize_Remap[0],(int)szGlobalWorkSize_Remap[0]/ (int)szLocalWorkSize_Remap[0]);
	    for (int y = 0; y < imLPyramid[l].h; ++y) {
	      	  	

	      	  	char profInfo[50];
		        SetRemappingSize( imLPyramid[l].w, hw, l ) ;
		        clock_gettime(CLOCK_MONOTONIC, &remapp_time_start);
		   //      if (l==0)
		   //      {
		   //      	#pragma acl task out(G_pyramid_address_space[0]) in(pim) workers(szLocalWorkSize_Remap[0]) groups(szGlobalWorkSize_Remap[0]) label("remapp_GPU0") bind(DEVICE_REMAPP)  
		   //  		remapp_GPU_level0( G_pyramid_address_space[0] ,pim,y, imLPyramid[l].w, imLPyramid[l].h,im.w,im.h,l , sigma,alpha_,beta_);
					// #pragma acl taskwait label("remapp_GPU0")
					// //remapp_GP( pyramid_address_space, pimPyramid[l] ,pim,y, imLPyramid[l].w, imLPyramid[l].h,im.w,im.h,l , sigma,alpha_,beta_,szLocalWorkSize_Remap[0],szGlobalWorkSize_Remap[0]/szLocalWorkSize_Remap[0]);

		   //  	}
		    	//else{
		    		pixel_t * gaussianP = imPyramid[l].img;

		    		#pragma acl task out(G_pyramid_address_space[0]) in(gaussianP) in(pim) workers(szLocalWorkSize_Remap[0]) groups(szGlobalWorkSize_Remap[0]) label("remapp_GPU") bind(DEVICE_REMAPP)  
		    		remapp_GPU( G_pyramid_address_space[0] ,gaussianP,pim,y, imLPyramid[l].w, imLPyramid[l].h,im.w,im.h,l , sigma,alpha_,beta_);
					#pragma acl taskwait label("remapp_GPU")
					//remapp_GP( pyramid_address_space, pimPyramid[l] ,pim,y, imLPyramid[l].w, imLPyramid[l].h,im.w,im.h,l , sigma,alpha_,beta_,szLocalWorkSize_Remap[0],szGlobalWorkSize_Remap[0]/szLocalWorkSize_Remap[0]);

		    	//}
		    	clock_gettime(CLOCK_MONOTONIC, &remapp_time_finish);
				remapp_time_elapsed += (remapp_time_finish.tv_sec - remapp_time_start.tv_sec);
				remapp_time_elapsed += (remapp_time_finish.tv_nsec - remapp_time_start.tv_nsec) / 1000000000.0;
				

				int yf = y*(1<<(l));
        		
          		int row_range_start = max(yf-hw,0);
            	int row_range_end =   min(yf+hw,im.h);

				

	   			Pyramid[0].w = 2*hw;//col_range.end - col_range.start;
				Pyramid[0].h = row_range_end - row_range_start;
 
		    	// Pyramid[0].w = 2*hw;
	    		// Pyramid[0].h = 2*hw;
		    	for (j = 1; j < l+2; j++) {
		    		Pyramid[j].w = Pyramid[j-1].w/2 +Pyramid[j-1].w%2;
		    		Pyramid[j].h = Pyramid[j-1].h/2 +Pyramid[j-1].h%2; 
		    		local_PyramidDown(  Pyramid[j-1].w, Pyramid[j-1].h, G_pyramid_address_space[j-1], Temp_pyramid_address_space[0], G_pyramid_address_space[j],  Buff_pyramid_address_space[0],y,l,j,hw,imLPyramid[l].w);
					//local_PyramidUp_sub(Pyramid[j-1].w, Pyramid[j-1].h, G_pyramid_address_space[j],   Temp_pyramid_address_space[0], L_pyramid_address_space[j-1],G_pyramid_address_space[j-1],Buff_pyramid_address_space[0],y,l,j,hw,imLPyramid[l].w);
        		}
        		local_PyramidUp_sub(Pyramid[l].w, Pyramid[l].h, G_pyramid_address_space[l+1],   Temp_pyramid_address_space[0], L_pyramid_address_space[l],G_pyramid_address_space[l],Buff_pyramid_address_space[0],y,l,l+1,hw,imLPyramid[l].w);

				//int yf = y*(1<<(l)) ;
				int yfc = yf-row_range_start;
				int yfclev0 = yfc>>l;
	      		for (int x = 0; x < imLPyramid[l].w; ++x) {

            		
            		int xf = x*(1<<(l)) ;
            		col_range.start = max(xf-hw,0);
            	    col_range.end =   min(xf+hw,im.w);
            	    int col_w = col_range.end- col_range.start;		
            	    int xfc = xf - col_range.start;
		       		int xfclev0 = xfc>>l;
		    		for (j = 1; j < l+1; j++) {
		    			col_w = col_w/2+ col_w%2;
        			}

		       		pixel_t value =G_pyramid_address_space[l][Pyramid[l].w*Pyramid[l].h*x + col_w*yfclev0 + xfclev0]- L_pyramid_address_space[l][Pyramid[l].w*Pyramid[l].h*x + col_w*yfclev0 + xfclev0];
  					imLPyramid[l].img[imLPyramid[l].w*y+x] =value;
	      		}
	    }
	    
		clock_gettime(CLOCK_MONOTONIC, &time_finish); 
		
		time_elapsed = (time_finish.tv_sec - time_start.tv_sec);
		time_elapsed += (time_finish.tv_nsec - time_start.tv_nsec) / 1000000000.0;

		time_elapsed_t[l]= time_elapsed;
		remapp_time_elapsed_t[l]= remapp_time_elapsed;
		blur_time_elapsed_t[l]= blur_time_elapsed;
		upsample_time_elapsed_t[l]= upsample_time_elapsed;
		downsample_time_elapsed_t[l]= downsample_time_elapsed;
		subtract_time_elapsed_t[l]= subtract_time_elapsed;

	}   
	clock_gettime(CLOCK_MONOTONIC, &time_finish); 
	//double time_elapsed;
	time_elapsed = (time_finish.tv_sec - time_start.tv_sec);
	time_elapsed += (time_finish.tv_nsec - time_start.tv_nsec) / 1000000000.0;
	printf("\n\n");
	printf("level %d \n\n",0);
	printf(">>>> level Time  = %f ms\n",time_elapsed_t[0]*1000);
	printf(">>>> Time remap = %f ms\n",remapp_time_elapsed_t[0]*1000);
	printf(">>>> Time blur  = %f ms\n",blur_time_elapsed_t[0]*1000);
	printf(">>>> Time upsample = %f ms\n",upsample_time_elapsed_t[0]*1000);
	printf(">>>> Time downsample = %f ms\n",downsample_time_elapsed_t[0]*1000);
	printf(">>>> Time subtract = %f ms\n",subtract_time_elapsed_t[0]*1000);

	for (int l = 1; l < J-1; l++)
	{
		printf("level %d \n\n",l);
		printf(">>>> level Time  = %f ms\n",time_elapsed_t[l]*1000-time_elapsed_t[l-1]*1000);
		printf(">>>> Time remap = %f ms\n",remapp_time_elapsed_t[l]*1000-remapp_time_elapsed_t[l-1]*1000);
		printf(">>>> Time blur  = %f ms\n",blur_time_elapsed_t[l]*1000-blur_time_elapsed_t[l-1]*1000);
		printf(">>>> Time upsample = %f ms\n",upsample_time_elapsed_t[l]*1000-upsample_time_elapsed_t[l-1]*1000);
		printf(">>>> Time downsample = %f ms\n",downsample_time_elapsed_t[l]*1000-downsample_time_elapsed_t[l-1]*1000);
		printf(">>>> Time subtract = %f ms\n",subtract_time_elapsed_t[l]*1000-subtract_time_elapsed_t[l-1]*1000);
	}
	printf("Total \n");

	printf(">>>> Total time  = %f ms\n",time_elapsed*1000);
	printf(">>>> Total time remap = %f ms\n",remapp_time_elapsed*1000);
	printf(">>>> Total time blur  = %f ms\n",blur_time_elapsed*1000);
	printf(">>>> Total time upsample = %f ms\n",upsample_time_elapsed*1000);
	printf(">>>> Total time downsample = %f ms\n",downsample_time_elapsed*1000);
	printf(">>>> Total time subtract = %f ms\n",subtract_time_elapsed*1000);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////  
    // Now collapse the output laplacian pyramid
    // printf("Collapsing laplacian pyramid down to output image\n");
    //printf("pyramids done and rescaled\n");

    for (j = J-1; j > 0; j--) {
		PyramidUp_add(imLPyramid[j-1].w, imLPyramid[j-1].h, imLPyramid[j].img,imTempPyramid[j-1].img , imLPyramid[j-1].img,imLPyramid[j-1].img,0,0);
    }
    //printf("pyramid colapsed\n");
    return imLPyramid[0].img;

}