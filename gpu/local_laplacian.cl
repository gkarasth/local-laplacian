#pragma OPENCL EXTENSION cl_khr_fp64 : enable



__kernel void convolutionRow_K(
    __global double *d_Result,
    __global double *d_Data,
    int dataW,
    int dataH
){
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    const double d_Kernel[5] = {1,5,8,5,1};
	int x, y, k;
	x= get_local_id(0);
    y= get_local_id(1);
    double sum = 0;
	if(x<dataW && y<dataH){
		for (k = -2; k <= 2; k++) {
			int d = x + k;

			if (d >= 0 && d < imageW) {
				sum += h_Src[ y * imageW + d] * d_Kernel[2 - k];
			}     

		}
		h_Dst[y * imageW + x] = sum/weight;
	}
}

__kernel void convolutionColumn_K(
    __global double *d_Result,
    __global double *d_Data,
    int dataW,
    int dataH
){
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
	const double d_Kernel[5] = {1,5,8,5,1};
	int x, y, k;
    x= get_local_id(0);
    y= get_local_id(1);
    double sum = 0;
	if (x<dataW && y <dataH)
	{
		for (k = -22; k <= 22; k++) {
			int d = y + k;

			if (d >= 0 && d < imageH) {
				sum += h_Src[d * imageW + x] * d_Kernel[22 - k];
			}   

		}
		h_Dst[y * imageW + x] = sum/(weight);
	}    
}





void Upsample_K(double * target , double * source,int imageH, int imageW,int c_off, int r_off,double * init){
    int x,y;
    x= get_local_id(0);
    y= get_local_id(1);
    if(x<imageW&& y < imageH){
        target[y*2*imageW*2 +2*x +c_off+(r_off*imageW*2)] = source[ y*imageW+x];
        target[(y*2+1)*imageW*2 +2*x+c_off+(r_off*imageW*2) ] = 0;   
        target[y*2*imageW*2 +2*x+1 +c_off+(r_off*imageW*2)] = 0;
        target[(y*2+1)*imageW*2 +2*x+1+c_off+(r_off*imageW*2)  ] = 0;
	}

}

void sub_K(double * target , double * Sed , double * Sor ,int imageH,int imageW){
    int x,y;
    x= get_local_id(0);
    y= get_local_id(1);
    if(x<imageW&& y < imageH){
            target[y*imageW+x] = Sed[y*imageW+x]-Sor[y*imageW+x];
        }
    }
}

void ConvBlur_K(int imageW, int imageH, double * img,double * target,double * h_Buff){

  
 
	convolutionRow_K(h_Buff,img,imageW,imageH);
  
	convolutionColumn_K(target,h_Buff,imageW,imageH);
 


}

void Downsample_K( double * target ,double * source,int imageH, int imageW,int c_off, int r_off){
    int x,y;
    x= get_local_id(0);
    y= get_local_id(1);
    if(x<imageW&& y < imageH){
        target[y*imageW +x ] = source[ y*imageW*4+x*2+(r_off*imageW*2)+c_off];  
    }
}

void PyramidDown_K(int imageW, int imageH, double * img, double * temp,double * target,int c_off, int r_off,double * buffer){
    ConvBlur_K( imageW,  imageH,  img, temp ,buffer);
    Downsample_K( target ,  temp, imageH/2,  imageW/2,c_off,r_off);
}

void PyramidUp_sub_K(int imageW, int imageH, double * img, double * temp,double * target,double * subtracted,int c_off, int r_off,double * buffer){

    Upsample_K(temp ,  img, imageH,  imageW,c_off,r_off,subtracted);
    ConvBlur_K( imageW*2,  imageH*2,  temp, target ,buffer);
    sub_K( target ,  subtracted ,  target , imageH*2, imageW*2);
}


double SmoothStep_K(double x_min, double x_max, double x) {
	#pragma OPENCL EXTENSION cl_khr_fp64 : enable
  double y = (x - x_min) / (x_max - x_min);
  y = max(0.0, min(1.0, y));
  return pow(y, 2) * pow(y-2, 2);
}

double DetailRemap_K(double delta,double alpha, double sigma_r ) {
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

double EdgeRemap_K(double delta,double beta) {
	#pragma OPENCL EXTENSION cl_khr_fp64 : enable

  return beta * delta;
}

void RemappingFunction_K(double value,
                                 double reference,
                                 double sigma_r,
                                 double alpha,
                                 double beta,
                                 int output_dst,
                                 __global double  * dst ) {
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
  	dst[output_dst] = temp;
  
}

__kernel void remapp_K(int Row_start ,int Col_start ,__global double  * dst ,__global double  * src,int srcW,int srcH,double reference ,double sigma,double alpha , double beta){
	int x,y;
	#pragma OPENCL EXTENSION cl_khr_fp64 : enable
	//int col_size = col_range.end - col_range.start;
	//int row_size = row_range.end - row_range.start;
	double input;

	x=get_local_id(0);
	y=get_local_id(1);

	
	if (y+Row_start>=srcH||x+Col_start>=srcW)
	{
		input=0;
	}
	else
	{
		input = src[ Col_start +x +(y+Row_start)*srcW];
	}
	int output_dst = x+y*get_global_size(0);
	RemappingFunction_K(input,reference,sigma,alpha,beta,output_dst,dst);
			
			
		
	
}

#define J 5 
__kernel void local_laplacian(
	int l,
	int kernel_space_size,
	int image_w,
	int image_h,
	int gaussian_w,
	int gaussian_h,
	double alpha,
	double beta,
	double sigma,
	__global double * image,
	__global double * gaussian,
	__global double * laplacian
	__global double * gp
	 ){
__local int Pyramid_w[J];
__local int Pyramid[J];
__local int LPyramid[J];
	int temp;
	int buffer;
	int y = get_group_id(0);
	int x = get_group_id(1);

	int Pyramid_base = (x+y*kernel_space_size)*kernel_space_size;

	int yf = y*(1<<(l)) ;
	int xf = x*(1<<(l)) ;


	r_start = max(0,yf-hw);
	r_end = min(kRows,yf+hw);

	c_start = max(0,xf-hw);
	c_end = min(kCols,xf+hw);


	Pyramid_w[0] = c_end - c_start;
	Pyramid_h[0] = r_end - r_start;
	Pyramid_w[0] +=Pyramid_w[0]%2;
	Pyramid_h[0] +=Pyramid_h[0]%2;


	for (i = 1; i < l+2; ++i)
	{	       				

		Pyramid_h[i] = Pyramid_h[i-1]/2;
		Pyramid_w[i] = Pyramid_w[i-1]/2;

		if (Pyramid_h[i]%2)
		{
			Pyramid_h[i]++;
			for (int j = i; j >=1; --j)
			{	
				Pyramid_h[j-1] = Pyramid_h[j]*2;
			}
		}

		if (Pyramid_w[i]%2)
			{
			Pyramid_w[i]++;
			for (int j = i; j >=1; --j)
			{	
				Pyramid_w[j-1] = Pyramid_w[j]*2;
			}
		}

	}

	temp = Pyramid_base;
	buffer = temp + Pyramid_h[0]*Pyramid_w[0];
	Pyramid[0]  = buffer + Pyramid_h[0]*Pyramid_w[0];
	
	for (i = 1; i < l+2; ++i)
	{	       				
			Pyramid[i]  =  Pyramid[i-1] + Pyramid_h[i-1]*Pyramid_w[i-1];
	}
	
	LPyramid[0].img = Pyramid[l+1].img + Pyramid[l+1].h*Pyramid[l+1].w;
	
	for (i = 1; i < l+2; ++i)
	{	 
			LPyramid[i] = LPyramid[i-1] + Pyramid_h[i-1]*Pyramid_w[i-1];
	}

	double g0 = gaussian[gaussian_w*y+x];
	remapp_K( r_start , c_start ,gp + Pyramid[0] ,image, image_w, image_h,g0 , sigma,alpha,beta);


	// bookkeeping to compute index of (lev0,y0,x0) within the
	// subwindow, at full-res and at current pyramid level
	int yfc = yf - r_start;
	int xfc = xf - c_start;
	int yfclev0 = yfc>>l;
	int xfclev0 = xfc>>l;
	cstart = c_start;
	rstart = r_start;	
	for (j = 1; j < l+2; j++) {
		int c_off=cstart%2;
		int r_off=rstart%2;
		cstart= cstart/2+c_off;
		rstart= rstart/2+r_off;
		PyramidDown_K(	gp+Pyramid_w[j-1], gp+Pyramid_h[j-1], gp+Pyramid[j-1], gp+temp , Pyramid[j],c_off,r_off,gp + buffer);
		PyramidUp_sub_K( 	gp+Pyramid_w[j]  , gp+Pyramid_h[j]  , gp+Pyramid[j]  , gp+temp , LPyramid[j-1] , pPyramid[j-1],c_off,r_off,gp + buffer);
	}

	laplacian[gaussian_w*y+x] = (pLPyramid[l]+gp)[Pyramid_w[l]*yfclev0 + xfclev0];

}