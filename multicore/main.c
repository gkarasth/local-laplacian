
#include <FreeImage.h>
#include <centaurus_acl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <getopt.h>
//#include "localLaplacian.h"
//#include  "Image.h"
//#include "fast_blur.h"
#include "readmap.h"
#include "paris_llf.h"
#include "Image.h"
#include "pgm.c"

//void run_cpu_gray_test(Image img_in, char *out_filename);


void run_cpu_gray_test(FULLImage* himage, Image img_in, char *out_filename)
{
    Image img_obuf,buffer;
    //img_obuf.img = (int*)malloc(4*img_in.w * img_in.h * sizeof(int));
    pixel_t *img_baseaddr = (pixel_t*)malloc(img_in.w * img_in.h * sizeof(pixel_t));
    memcpy(img_baseaddr,img_in.img,img_in.w * img_in.h* sizeof(pixel_t));

    //int off = LocalLaplacian(img_in.w,img_in.h,img_baseaddr,4,1);
    pixel_t * off = paris_llf(img_in.w,img_in.h,img_baseaddr,0.25, 1,0.4);
     
    //printf("offset is : %d\n",off);
    img_obuf.w = img_in.w;
    img_obuf.h =img_in.h;
    img_obuf.img  = off;
    printf("Running local laplacian filter for gray-scale images.\n");
    //write_pgm(himage,img_obuf,out_filename);
    //write_pgm(img_obuf,"1.pgm");
    FULLImagewrite(himage,img_obuf,out_filename);

    //write_pgm(img_obuf, out_filename);
    //free_pgm(img_obuf);
    //free_pgm(img_in);
    free(img_baseaddr);
    //write_pgm(img_obuf, out_filename);
}





FULLImage * himage;

int main(int argc, char *argv[]){
    Image img_ibuf_g;

	if (argc != 3) {
		printf("Run with input file name and output file name as arguments\n");
		exit(1);
	}
	

    FULLImage * himage;

    char* inputImagePath = argv[1];
    himage = (FULLImage*)malloc(sizeof(FULLImage));
    Image * buffimage;
    buffimage = (Image *)malloc(sizeof(Image));
    FULLImageF(himage,inputImagePath,buffimage);
    img_ibuf_g = resize_image(buffimage);

    //FULLImagewrite(himage,testimage);

    printf("Running local laplacian filter for gray-scale images.\n");
    //img_ibuf_g = read_pgm(argv[1]);
    run_cpu_gray_test(himage,img_ibuf_g, argv[2]);
    //free_pgm(img_ibuf_g);
    //write_pgm(*testimage,argv[2]);
	return 0;
}




