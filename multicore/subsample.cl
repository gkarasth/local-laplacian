#include "Image.h"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void downsampleGPU(
    __global pixel_t *d_Result,
    __global pixel_t *d_Input,
    int y_offset, 
    int InputW,
    int InputH,
    int l,
    int j,
    int hw
){

    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
  

  

  int cstart = get_group_id(0)*(1<<(l)) - hw;
  int rstart = y_offset*(1<<(l)) -hw;
  int r_off;
  int c_off;
  for (int i = 1; i <= j; ++i)
  {
    c_off=abs(cstart%2);
    r_off=abs(rstart%2);
    cstart= cstart/2+c_off;
    rstart= rstart/2+r_off;
  }

  int ResultH = InputH/2+InputH%2;
  int ResultW = InputW/2+InputW%2;
  int write_offset = get_group_id(0)*ResultW*ResultH;
  int read_offset = get_group_id(0)*InputW*InputH+c_off+r_off*InputW;
  int x = get_local_id(0);
  int y = get_group_id(1);
  if(2*x + c_off>=InputW || 2*y + r_off>=InputH){
    d_Result[write_offset +x +y*ResultW ] = 0;
  }
  else{
    d_Result[write_offset +x +y*ResultW ] = d_Input[read_offset + 2*x +2*y*InputW];
  }

  
}
