
#define TILE_W 300

#ifndef max
  #define max(a, b) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
  #define min(a, b) ( ((a) < (b)) ? (a) : (b) )
#endif

double SmoothStep_GP(double x_min, double x_max, double x) {
  double y = (x - x_min) / (x_max - x_min);
  y = max(0.0, min(1.0, y));
  return pow(y, 2) * pow(y-2, 2);
}

double DetailRemap_GP(double delta,double alpha, double sigma_r ) {
  double fraction = delta / sigma_r;
  double polynomial = pow(fraction, alpha);
  if (alpha < 1) {
    const double kNoiseLevel = 0.01;
    double blend = SmoothStep_GP(kNoiseLevel,2 * kNoiseLevel, fraction * sigma_r);
    polynomial = blend * polynomial + (1 - blend) * fraction;
  }
  return polynomial;
}

double EdgeRemap_GP(double delta,double beta) {

  return beta * delta;
}

void RemappingFunction_GP(double value,
                                 double reference,
                                 double sigma_r,
                                 double alpha,
                                 double beta,
                                 int output_dst,
                                  double  * dst ) {
  	double delta = reference - value;
  	if(value>reference)
  		delta = value - reference;
  	
  	int sign = value < reference ? -1 : 1;
  	
	double temp;
  	if (delta < sigma_r) {
    	temp = reference + sign * sigma_r * DetailRemap_GP(delta,alpha, sigma_r);     	

  	} else {
    	temp = reference + sign * (EdgeRemap_GP(delta - sigma_r,beta) + sigma_r);	
  	}
  	dst[output_dst] = temp;
  
}



 void remapp_GP(
    double *d_Result,
    double *d_Gaussian,
    double *d_Image,
    int y_offset, 
    int GaussianW,
    int GaussianH,
    int imageW,
    int imageH, 
    int l,
    double alpha,
    double beta,
    double sigma,
    int local_size0,
    int group_size1
){

   
  double data[TILE_W];
  

  
  int current_job = 0;
 int local_id;
  for (int group_id = 0; group_id < group_size1; ++group_id)
  {
        int global_y = y_offset;

        int hw = 3*(1<<(l+1))-2;
        int yf = global_y*(1<<(l));
        int row_range_start = yf-hw;
        int row_range_end = yf+hw;
        int subimage_size = hw*hw*4;
    for (int y = row_range_start; y < row_range_end; ++y)
    {
      for (local_id = 0; local_id < local_size0; ++local_id)
      {
        
        int x = local_id;       
        int local_work_size = (local_size0 - 2*hw)/(1<<(l)); //each block creates this amount of tone mapped sub images
        int global_x_offset  = local_work_size*group_id ;  //x on the gaussian 
        int image_x = global_x_offset*(1<<(l)) + x;               //map x on image coordinates
        int load_pos = image_x-hw;                                //every thread loads from image position 


        if(load_pos<0||load_pos>=imageW||y<0||y>=imageH)
        {
          data[x]=0;
        }
        else{
          data[x] = d_Image[load_pos+y*imageW];
        }
      }
      
      for (local_id = 0; local_id < local_size0; ++local_id)
      {  
            int x = local_id;       
            int local_work_size = (local_size0 - 2*hw)/(1<<(l)); //each block creates this amount of tone mapped sub images
            int global_x_offset  = local_work_size*group_id ;  //x on the gaussian 
            int image_x = global_x_offset*(1<<(l)) + x;        //map x on image coordinates
            current_job =0;
            while(current_job <= local_work_size){
              if(x<2*hw){
                double g0 = d_Gaussian[global_y*GaussianW+global_x_offset + current_job];
                double input = data[current_job*(1<<(l)) + x];

                int row_addres = (y-row_range_start)*2*hw;
                
                int subimage_addr = subimage_size*(global_x_offset + current_job);
               
                int output_dst = subimage_addr + x + row_addres;
                RemappingFunction_GP(input,g0,sigma,alpha,beta,output_dst,d_Result);
              }
              current_job++;
            }
      }

    }
  }
}