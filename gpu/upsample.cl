#include "Image.h"

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void UpsampleGPU(
    __global pixel_t *d_Result,
    __global pixel_t *d_Input,
    int y_offset, 
    int ResultW,
    int ResultH,
    int l,
    int j,
    int hw
){
  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
  int cstart= get_group_id(0)*(1<<(l)) - hw;
  int rstart = y_offset*(1<<(l)) -hw;
  if (rstart<0)rstart =0;

  int InputH = get_global_size(1);
  int InputW = get_local_size(0);
  int r_off;
  int c_off;
  for (int i = 1; i <= j; ++i)
  {
    c_off=abs(cstart%2);
    r_off=abs(rstart%2);
    cstart= cstart/2+c_off;
    rstart= rstart/2+r_off;
  }


  int x = get_local_id(0);
  int y = get_group_id(1);

  int write_offset = get_group_id(0)*ResultW*ResultH;
  int read_offset = get_group_id(0)*InputW*InputH;

  int addr0x = 2*x + c_off ;
  int addr0y = 2*y + r_off ;

  int addr1x = 2*x + 1 - c_off;  
  int addr1y = 2*y + r_off ;

  int addr2x = 2*x + c_off;
  int addr2y = 2*y + 1 - r_off; 

  int addr3x = 2*x + 1 - c_off;
  int addr3y = 2*y + 1 - r_off; 

  if(addr0y<ResultH && addr0x<ResultW)
    d_Result[write_offset+addr0x+addr0y*ResultW]=4.0*d_Input[read_offset+x+y*InputW];
  if(addr1y<ResultH && addr1x<ResultW)
    d_Result[write_offset+addr1x+addr1y*ResultW]=0;
  if(addr2y<ResultH && addr2x<ResultW)
    d_Result[write_offset+addr2x+addr2y*ResultW]=0;
  if(addr3y<ResultH && addr3x<ResultW)
    d_Result[write_offset+addr3x+addr3y*ResultW]=0;
}

//56381.417460
//25870.174333 