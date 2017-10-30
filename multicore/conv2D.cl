#ifndef __CONV2D_CL__
#define __CONV2D_CL__
#include "fast_blur.h"
#include "Image.h"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable


#define WEIG 20;

#define kernelR 2
#define kernelW (2 * kernelR + 1)
//__device__ __constant__ float d_Kernel[KERNEL_W];


#define            ROW_TILE_W 128
#define KERNEL_RADIUS_ALIGNED 16


#define COLUMN_TILE_W 16
#define COLUMN_TILE_H 48

#define IMUL(a, b) mul24((int)a, (int)b)


__kernel void convolutionRowGPU(
    __global pixel_t *d_Result,
    __global pixel_t *d_Data,
    int dataW,
    int dataH
){
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    const pixel_t d_Kernel[5] = {1,5,8,5,1};

    //Data cache
    __local pixel_t data[kernelR + ROW_TILE_W + kernelR];

    //Current tile and apron limits, relative to row start
    const int         tileStart = IMUL(get_group_id(0), ROW_TILE_W);
    const int           tileEnd = tileStart + ROW_TILE_W - 1;
    const int        apronStart = tileStart - kernelR;
    const int          apronEnd = tileEnd   + kernelR;

    //Clamp tile and apron limits by image borders
    const int    tileEndClamped = min(tileEnd, dataW - 1);
    const int apronStartClamped = max(apronStart, 0);
    const int   apronEndClamped = min(apronEnd, dataW - 1);
    const int          rowStart = IMUL(get_group_id(1), dataW);
    const int apronStartAligned = tileStart - KERNEL_RADIUS_ALIGNED;

    const int loadPos = apronStartAligned + get_local_id(0);
    //Set the entire data cache contents
    //Load global memory values, if indices are within the image borders,
    //or initialize with zeroes otherwise
    if(loadPos >= apronStart){
        const int smemPos = loadPos - apronStart;

        data[smemPos] = 
            ((loadPos >= apronStartClamped) && (loadPos <= apronEndClamped)) ?
            d_Data[rowStart + loadPos] : 0;
    }


    barrier(CLK_LOCAL_MEM_FENCE);
    const int writePos = tileStart + get_local_id(0);

    if(writePos <= tileEndClamped){
        const int smemPos = writePos - apronStart;
        pixel_t sum = 0;


        for(int k = -kernelR; k <= kernelR; k++)
            sum += data[smemPos + k] * d_Kernel[kernelR - k];

        d_Result[rowStart + writePos] = sum/WEIG;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__kernel void convolutionColumnGPU(
    __global pixel_t *d_Result,
    __global pixel_t *d_Data,
    int dataW,
    int dataH,
    int smemStride,
    int gmemStride
){
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable

    const pixel_t d_Kernel[5] = {1,5,8,5,1};

    //Data cache
    __local pixel_t data[COLUMN_TILE_W * (kernelR + COLUMN_TILE_H + kernelR)];

    //Current tile and apron limits, in rows
    const int         tileStart = IMUL(get_group_id(1), COLUMN_TILE_H);
    const int           tileEnd = tileStart + COLUMN_TILE_H - 1;
    const int        apronStart = tileStart - kernelR;
    const int          apronEnd = tileEnd   + kernelR;

    //Clamp tile and apron limits by image borders
    const int    tileEndClamped = min(tileEnd, dataH - 1);
    const int apronStartClamped = max(apronStart, 0);
    const int   apronEndClamped = min(apronEnd, dataH - 1);

    //Current column index
    const int       columnStart = IMUL(get_group_id(0), COLUMN_TILE_W) + get_local_id(0);

    //Shared and global memory indices for current column
    int smemPos = IMUL(get_local_id(1), COLUMN_TILE_W) + get_local_id(0);
    int gmemPos = IMUL(apronStart + get_local_id(1), dataW) + columnStart;
    //Cycle through the entire data cache
    //Load global memory values, if indices are within the image borders,
    //or initialize with zero otherwise
    for(int y = apronStart + get_local_id(1); y <= apronEnd; y += get_local_size(1)){
        data[smemPos] = ((y >= apronStartClamped) && (y <= apronEndClamped)) ? d_Data[gmemPos] : 0;
        smemPos += smemStride;
        gmemPos += gmemStride;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    //Shared and global memory indices for current column
    smemPos = IMUL(get_local_id(1) + kernelR, COLUMN_TILE_W) + get_local_id(0);
    gmemPos = IMUL(tileStart + get_local_id(1) , dataW) + columnStart;
    //Cycle through the tile body, clamped by image borders
    //Calculate and output the results
    for(int y = tileStart + get_local_id(1); y <= tileEndClamped; y += get_local_size(1)){
        pixel_t sum = 0;


        for(int k = -kernelR; k <= kernelR; k++)
            sum += 
                data[smemPos + IMUL(k, COLUMN_TILE_W)] *
                d_Kernel[kernelR - k];


        d_Result[gmemPos] = sum/WEIG;
        smemPos += smemStride;
        gmemPos += gmemStride;
    }
}




__kernel void convolutionRowGPUlocal(
    __global pixel_t *d_Result,
    __global pixel_t *d_Data,
    int dataW,
    int dataH,
    int hw,
    int l,
    int j,
    int ImageW,
    int num_of_images
){
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    //const pixel_t d_Kernel[5] = {1,5,8,5,1};
    const pixel_t d_Kernel[5] = {0.05, 0.25, 0.4, 0.25, 0.05};
    const int global_job = get_group_id(0);

    int cstart = global_job*(1<<(l)) - hw;
    int cend =   global_job*(1<<(l)) + hw;
      if (cstart<0) cstart = 0;         //max
      if (cend >ImageW) cend = ImageW;  //min 
    
    int RealW= cend - cstart;
    for (int i = 1; i < j; ++i)
    {
        RealW = RealW/2 + RealW%2;
    }
    //RealW = dataW;

    //Data cache
    __local pixel_t data[kernelR + ROW_TILE_W + kernelR];
    const int loadpos = get_local_id(0)-2;
    const int gmemPos = loadpos+get_group_id(1)*RealW;
    const int image_size = dataH*dataW;
    data[get_local_id(0)] = (loadpos>=0 && loadpos<RealW) ? d_Data[global_job*image_size + gmemPos] : 0;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    const int smemPos = get_local_id(0)+2;
    if(get_local_id(0) < RealW){
            pixel_t sum = 0;
            for(int k = -kernelR; k <= kernelR; k++)
                sum += data[smemPos + k] * d_Kernel[kernelR - k];

            d_Result[global_job*image_size +get_local_id(0) + get_group_id(1)*RealW ]= sum;
    }
    
}



__kernel void convolutionColumnGPUlocal(
    __global pixel_t *d_Result,
    __global pixel_t *d_Data,
    int dataW,
    int dataH,
    int hw,
    int l,
    int j,
    int ImageW,
    int num_of_images
){
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable

    //const pixel_t d_Kernel[5] = {1,5,8,5,1};
    const pixel_t d_Kernel[5] = {0.05, 0.25, 0.4, 0.25, 0.05};
    
    const int global_job = get_group_id(0);
    
    int cstart = global_job*(1<<(l)) - hw;
    int cend =   global_job*(1<<(l)) + hw;
      if (cstart<0) cstart = 0;         //max
      if (cend >ImageW) cend = ImageW;  //min 
    
    int RealW = cend - cstart;
    for (int i = 1; i < j; ++i)
    {
        RealW = RealW/2 + RealW%2;
    }


    __local pixel_t data[kernelR + ROW_TILE_W + kernelR];

        const int loadpos = get_local_id(0)-2;
        const int gmemPos = loadpos*RealW+get_group_id(1);
        
        const int image_size = dataH*dataW;
        data[get_local_id(0)] = (loadpos>=0 && loadpos<dataH) ? d_Data[global_job*image_size + gmemPos] : 0;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        const int smemPos = get_local_id(0)+2;
        if(get_local_id(0)< dataH && get_group_id(1)<RealW){
                pixel_t sum = 0;
                for(int k = -kernelR; k <= kernelR; k++)
                    sum += data[smemPos + k] * d_Kernel[kernelR - k];

                d_Result[global_job*image_size +get_local_id(0)*RealW +get_group_id(1) ]= sum;
        }
    }
#endif