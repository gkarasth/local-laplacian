#include "Image.h"
#define R_TILE_W 220

#define min_i( a,  b) (a > b) ? (b) : (a);


#define max_i( a,  b) (a > b) ? (a) : (b);

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
double SmoothStep_GPU(double x_min, double x_max, double x) {
	#pragma OPENCL EXTENSION cl_khr_fp64 : enable
  double y = (x - x_min) / (x_max - x_min);
  y = max(0.0, min(1.0, y));
  return pow(y, 2) * pow(y-2, 2);
}

double DetailRemap_GPU(double delta,double alpha, double sigma_r ) {
	#pragma OPENCL EXTENSION cl_khr_fp64 : enable
  double fraction = delta / sigma_r;
  double polynomial = pow(fraction, alpha);
  if (alpha < 1) {
    const double kNoiseLevel = 0.01;
    double blend = SmoothStep_GPU(kNoiseLevel,2 * kNoiseLevel, fraction * sigma_r);
    polynomial = blend * polynomial + (1 - blend) * fraction;
  }
  return polynomial;
}

double EdgeRemap_GPU(double delta,double beta) {
	#pragma OPENCL EXTENSION cl_khr_fp64 : enable

  return beta * delta;
}

void RemappingFunction_GPU(double value,
                                 double reference,
                                 double sigma_r,
                                 double alpha,
                                 double beta,
                                 int output_dst,
                                 __global pixel_t  * dst ) {
	#pragma OPENCL EXTENSION cl_khr_fp64 : enable
  	double delta = reference - value;
  	if(value>reference)
  		delta = value - reference;
  	
  	int sign = value < reference ? -1 : 1;
  	
	double temp;
  	if (delta < sigma_r) {
    	temp = reference + sign * sigma_r * DetailRemap_GPU(delta,alpha, sigma_r);     	

  	} else {
    	temp = reference + sign * (EdgeRemap_GPU(delta - sigma_r,beta) + sigma_r);	
  	}
  	dst[output_dst] = (pixel_t)temp;
  
}


__kernel void remapp_GPU(
    __global pixel_t *d_Result,
    __global pixel_t *d_Gaussian,
    __global pixel_t *d_Image,
    int global_y, 
    int GaussianW,
    int GaussianH,
    int imageW,
    int imageH, 
    int l,
    double alpha,
    double beta,
    double sigma
){

    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
  __local pixel_t sh_data[R_TILE_W];
  
  int tid = get_local_id(0);
  int hw = 3*(1<<(l+1))-2;

  int local_job_size = (get_local_size(0) - 2*hw)/(1<<(l)); //each block-group creates this amount of tone mapped sub images
  int global_x_offset  = local_job_size*get_group_id(0) ;   //x on the gaussian 

  int threadperjob = (get_local_size(0)/local_job_size);
  int job = tid/threadperjob;
  

  int global_x = global_x_offset +job;
  pixel_t g0 = d_Gaussian[global_y*GaussianW+global_x];
  

  int load_pos = global_x_offset*(1<<l) - hw + tid;

  int yf = global_y*(1<<(l));
  // int row_range_start = yf-hw;
  // int row_range_end = yf+hw;
  int row_range_start = (yf-hw > 0) ? (yf-hw) : (0);            //max_i(yf-hw,0);
  int row_range_end =   (yf+hw < imageH) ? (yf+hw) : (imageH);  //min_i(yf+hw,imageH);

    int xf = global_x*(1<<l);
    int col_range_start = (xf-hw > 0) ? (xf-hw) : (0);          //xf - hw;//
    int col_range_end   = (xf+hw < imageW) ? (xf+hw) : (imageW);//xf + hw;//
    // int col_range_start = xf - hw;
    // int col_range_end   = xf + hw;
    int col_size = col_range_end - col_range_start;

    int subimage_size = 2*hw*(row_range_end-row_range_start);
    int local_cs = col_range_start - (xf - hw);
    int subimage_addr = subimage_size*global_x;
  
  for (int y = row_range_start; y < row_range_end; ++y)
  {
    if(load_pos<0||load_pos>=imageW)
    {
      sh_data[tid]=0;
    }
    else{
      sh_data[tid] = d_Image[load_pos+y*imageW];
    }

    barrier(CLK_LOCAL_MEM_FENCE);




    int row_addres = (y-row_range_start)*col_size;

    int xpos = tid%threadperjob;
    if(job<local_job_size){
      while (xpos<col_size)
      {
        pixel_t input = sh_data[local_cs + xpos+job*(1<<l)];

        int output_dst =subimage_addr + xpos + row_addres;
        RemappingFunction_GPU((double)input,(double)g0,sigma,alpha,beta,output_dst,d_Result);
        //d_Result[output_dst] = input;;
        xpos += threadperjob;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}


__kernel void remapp_GPU_level0(
    __global pixel_t *d_Result,
    __global pixel_t *d_Image,
    int global_y, 
    int GaussianW,
    int GaussianH,
    int imageW,
    int imageH, 
    int l,
    double alpha,
    double beta,
    double sigma
){

    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
  __local pixel_t sh_data[R_TILE_W];
  
  int tid = get_local_id(0);
  int hw = 3*(1<<(l+1))-2;

  int local_job_size = (get_local_size(0) - 2*hw)/(1<<(l)); //each block-group creates this amount of tone mapped sub images
  int global_x_offset  = local_job_size*get_group_id(0) ;   //x on the gaussian 

  int threadperjob = (get_local_size(0)/local_job_size);
  int job = tid/threadperjob;
  

  int global_x = global_x_offset +job;
  pixel_t g0 = d_Image[global_y*GaussianW+global_x];
  

  int load_pos = global_x_offset*(1<<l) - hw + tid;

  int yf = global_y*(1<<(l));
  // int row_range_start = yf-hw;
  // int row_range_end = yf+hw;
  int row_range_start = (yf-hw > 0) ? (yf-hw) : (0);            //max_i(yf-hw,0);
  int row_range_end =   (yf+hw < imageH) ? (yf+hw) : (imageH);  //min_i(yf+hw,imageH);

    int xf = global_x*(1<<l);
    int col_range_start = xf - hw;
    int col_range_end   = xf + hw;
    int col_size = col_range_end - col_range_start;
    int subimage_size =2*hw*(row_range_end-row_range_start); //col_size*(row_range_end-row_range_start);
    int local_cs = col_range_start - (xf - hw);
    int subimage_addr = subimage_size*global_x;
  
  for (int y = row_range_start; y < row_range_end; ++y)
  {
    if(load_pos>=0&&load_pos<imageW)
    {
      sh_data[tid]=0;
    }
    else{
      sh_data[tid] = d_Image[load_pos+y*imageW];
    }

    barrier(CLK_LOCAL_MEM_FENCE);




    int row_addres = (y-row_range_start)*col_size;

    int xpos = tid%threadperjob;
    if(job<local_job_size){
      while (xpos<col_size)
      {
        pixel_t input = sh_data[local_cs + xpos+job*(1<<l)];

        int output_dst =subimage_addr + xpos + row_addres;
        RemappingFunction_GPU((double)input,(double)g0,sigma,alpha,beta,output_dst,d_Result);
        xpos += threadperjob;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
