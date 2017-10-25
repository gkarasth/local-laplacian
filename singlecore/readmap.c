#include "readmap.h"
#include "readmap.h"
#include <stdlib.h>
#include <stdio.h>
#include <FreeImage.h>
#include <math.h>
//#include "Image.h"
unsigned char  *global;
mapper *rgbmap;
    
    void FULLImageF(FULLImage* inst,char* fileName,Image *inputimg)
    {
        int bLoaded = 0;

        FIBITMAP *bmp = 0;
        FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
        fif = FreeImage_GetFileType(fileName,0);
        if (fif == FIF_UNKNOWN)
        {
            fif = FreeImage_GetFIFFromFilename(fileName);
            printf("FIF_UNKNOWN\n");
        }

        if (fif != FIF_UNKNOWN && FreeImage_FIFSupportsReading(fif))
        {
            bmp = FreeImage_Load(fif, fileName, 0);
            bLoaded = 1;
            if (bmp == NULL)
                bLoaded = 0;
        }
        FIBITMAP *bmpTemp;
        if (bLoaded)
        {
            inst->width = FreeImage_GetWidth(bmp);
            inst->height = FreeImage_GetHeight(bmp);

            inst->bpp = FreeImage_GetBPP(bmp);
            switch (inst->bpp)
            {
            case 32:
                break;
            default:

                bmpTemp = FreeImage_ConvertTo32Bits(bmp);
                if (bmp != NULL) FreeImage_Unload(bmp);
                bmp = bmpTemp;
                inst->bpp = FreeImage_GetBPP(bmp);
                break;
            }

            inst->pixels = (unsigned char*) malloc(sizeof(unsigned char) * 4 * inst->width * inst->height);
            global = inst->pixels;
            FreeImage_ConvertToRawBits(inst->pixels, bmp, inst->width * 4, inst->bpp, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, 1);
            /////
            int x,y;
            double temp_pxl;
            inputimg->w = inst->width;
            inputimg->h = inst->height;
            inputimg->img = (pixel_t *)malloc(inputimg->w * inputimg->h * sizeof(pixel_t));
            rgbmap = (mapper *) malloc(sizeof(mapper)*inputimg->w*inputimg->h);

            //inst->pixels[4*i] );//B*1
            //inst->pixels[4*i+1] );//G*40           
            //inst->pixels[4*i+2] );//R*20
           
           
          
            for (y = 0; y <inputimg->h; ++y)
            {
                for (x = 0; x <inputimg->w; ++x)
                {   
                    //img_ibuf = (1/61)(20Ir+40Ig+Ib)
                    temp_pxl = (inst->pixels[4*(x + inputimg->w*y)]+40.0*inst->pixels[4*(x + inputimg->w*y)+1]+20*inst->pixels[4*(x + inputimg->w*y)+2])/61;
                    
                    //temp_pxl+=(double)EPSILON;
                    inputimg->img[x+inputimg->w*y]=(pixel_t)temp_pxl/255;//(x)*100+y;//
                    //if(temp_pxl==0)
                        
                        //printf("%d %d %d %d \n",inst->pixels[4*(x + inputimg->w*y)],inst->pixels[4*(x + inputimg->w*y)+1],inst->pixels[4*(x + inputimg->w*y)+2],inst->pixels[4*(x + inputimg->w*y)+3]);
                        //(ρr,ρg,ρb) =(1/Ii)(Ir,Ig,Ib)                  
                        rgbmap[x+inputimg->w*y].B = ( double )inst->pixels[4*(x + inputimg->w*y)  ]/temp_pxl;
                        rgbmap[x+inputimg->w*y].G = ( double )inst->pixels[4*(x + inputimg->w*y)+1]/temp_pxl;
                        rgbmap[x+inputimg->w*y].R = ( double )inst->pixels[4*(x + inputimg->w*y)+2]/temp_pxl;
                    
                }
            
            }
            // for (int i = 0; i < 10; ++i)
            // {
            //     printf("%f  ",inputimg->img[i] );
            // }
            
            ////
            inst->isLoaded = 1;
        }
        else
            inst->isLoaded = 0;


        
    }
    Image resize_image(Image *inputimg)
    {

        pixel_t *temp;
        int i,y,x,tempw,temph;
        Image result;
        int v_max;//, i;

        result.w = inputimg->w;
        result.h = inputimg->h;
        printf("Image size: %d x %d\n", result.w, result.h);
        
       
        temp = inputimg->img;//(unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

        tempw = result.w;
        temph = result.h;

        result.w = result.w + result.w%2;
        result.h = result.h + result.h%2;
        // for (i = 0; i < 8; ++i)
        // {
        //     result.w/=2;
        //     result.h/=2;
        // }
        // result.h *=256;
        // result.w *=256;
        // if (result.h!=temph)
        //     result.h +=256;

        // if (result.w!=tempw)
        // result.w +=256;

        //result = newImage(result.w,result.h);
        //result = newImage(result.w,result.h);
        result.img = (pixel_t *)malloc(result.w * result.h * sizeof(pixel_t));
        printf("converted to : %d x %d\n", result.w, result.h);

       
        for (y = 0; y <result.h; ++y)
        {
            for (x = 0; x <result.w; ++x)
            { 
                result.img[x+result.w*y] = 0;
            }
        }

        for (y = 0; y <result.h; ++y)
        {
            for (x = 0; x <result.w; ++x)
            {
                    if (x>=tempw || y>=temph)
                    {
                        result.img[x+result.w*y] = 0;
                    }
                    else
                    {
                        result.img[x+result.w*y] = (pixel_t)temp[x + tempw*y];
                    }
            }
            
        }
 
        // for (y = 0; y <temph; ++y)
        // {
        //     for (x = 0; x <tempw; ++x)
        //     { 
        //         result.img[100 +100*result.w+x+result.w*y] = (pixel_t)temp[x + tempw*y];
        //     }
        // }


        free(temp);
        printf("converted to : %d x %d\n", result.w, result.h);

        return result;
    }
	
   void FULLImagewrite(FULLImage* img,Image inputimg,char *out_filename){
        FIBITMAP *bmp = 0;
        int x,y;
        double temp_pxl;

        for (y = 0; y <img->height; ++y)
        {

            for (x = 0; x <img->width; ++x)
            {   
                temp_pxl = (double)inputimg.img[x+inputimg.w*y]*255;

                double temp_B=(temp_pxl*rgbmap[x + img->width*y].B);
                if(temp_B>255)
                    temp_B= 255;
                if(temp_pxl<0)
                    temp_B =0;
                img->pixels[4*(x + img->width*y) ] = (unsigned char)temp_B;

                double temp_G=(temp_pxl*rgbmap[x + img->width*y].G);
                if(temp_G>255)
                    temp_G= 255;
                if(temp_G<0)
                    temp_G =0;
                img->pixels[4*(x + img->width*y)+1] = (unsigned char)temp_G;
                
                double temp_R=(temp_pxl*rgbmap[x + img->width*y].R);
                if(temp_R>255)
                    temp_R= 255;
                if(temp_R<0)
                    temp_R =0;
                img->pixels[4*(x + img->width*y)+2] = (unsigned char)temp_R;
                
                img->pixels[4*(x + img->width*y)+3] = 255;


            }
            
        }

        bmp = FreeImage_ConvertFromRawBits(img->pixels,img->width, img->height, img->width * 4, img->bpp, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, 1);
        FreeImage_Save(FIF_PNG, bmp, out_filename, 0);
    }