#include "Image.h"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void Sub_GPU(
    __global pixel_t *d_Result,
    __global pixel_t *d_Input
)
{
  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
 int pos = get_local_size(0)*get_group_id(0) + get_local_id(0);
 d_Result[pos]=  d_Input[pos] - d_Result[pos];
}