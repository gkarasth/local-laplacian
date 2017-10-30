#include "Image.h"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable


__kernel void downsampleGPU(
    __global pixel_t *d_Result,
    __global pixel_t *d_Input,
    int y_offset, 
    int InputW,
    int InputH,
    int ImageW,
    int l,
    int j,
    int hw
){

    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
  


  int cstart = get_group_id(0)*(1<<(l)) - hw;//(xf-hw > 0) ? (xf-hw) : (0); 
  int cend =   get_group_id(0)*(1<<(l)) + hw;//(xf+hw < imageW) ? (xf+hw) : (imageW);
  
  if (cstart<0) cstart = 0;         //max
  if (cend >ImageW) cend = ImageW;  //min 

  int RealResultW =cend - cstart;  
  int RealInputW = RealResultW*2;

  int rstart = y_offset*(1<<(l)) -hw;
  
  if (rstart<0) rstart = 0;
  int r_off;
  int c_off;
  for (int i = 1; i <= j; ++i)
  {
    c_off=abs(cstart%2);
    r_off=rstart%2;
    cstart= cstart/2+c_off;
    rstart= rstart/2+r_off;
    RealInputW = RealResultW;
    RealResultW = RealResultW/2 + RealResultW%2;
  }

  int ResultH = InputH/2+InputH%2;
  int ResultW = InputW/2+InputW%2;

  // RealResultW = ResultW;
  // RealInputW  = InputW;
  int write_offset = get_group_id(0)*ResultW*ResultH;
  int read_offset = get_group_id(0)*InputW*InputH+c_off+r_off*RealInputW;
  int x = get_local_id(0);
  int y = get_group_id(1);
  if(x<RealResultW){
    if(2*x + c_off>=RealInputW || 2*y + r_off>=InputH){
      d_Result[write_offset +x +y*RealResultW ] = 0;
    }
    else{
      d_Result[write_offset +x +y*RealResultW ] = d_Input[read_offset + 2*x +2*y*RealInputW];
    }
  }

  
}