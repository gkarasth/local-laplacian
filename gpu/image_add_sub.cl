#include "Image.h"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void Sub_GPU(
    __global pixel_t *d_L,
    __global pixel_t *d_G,
    __global pixel_t *d_Result,
    int l,
    int hw,
    int Input_w,
    int Input_h
)
{
  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
	int yfclev0 = hw>>l;
	int xfclev0 = hw>>l;
	int size = Input_h*Input_w;
 int pos = get_local_size(0)*get_group_id(0) + get_local_id(0);
 d_Result[pos]=  d_G[size*pos+xfclev0+yfclev0*Input_w] - d_L[size*pos+xfclev0+yfclev0*Input_w];
}

//pixel_t value = L_pyramid_address_space[l][Pyramid[l].w*Pyramid[l].h*x + Pyramid[l].w*yfclev0 + xfclev0];