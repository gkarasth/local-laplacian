
// int tempw,temph;

// Image read_pgm(const char * path){
//     FILE * in_file;
//     char sbuf[256];
//     unsigned char *temp;
//     int i,y,x;
//     Image result;
//     int v_max;//, i;
//     in_file = fopen(path, "r");
//     if (in_file == NULL){
//         printf("Input file not found!\n");
//         exit(1);
//     }
//     //char buffer[10];
//     fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
//     //fscanf(in_file, "%[^\n]\n",buffer);
//     fscanf(in_file, "%d",&result.w);
//     fscanf(in_file, "%d",&result.h);
//     fscanf(in_file, "%d\n",&v_max);
//     printf("Image size: %d x %d\n", result.w, result.h);
    
//    //result.img = (int *)malloc(result.w * result.h * sizeof(int));
   
//     temp = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

//     fread(temp,sizeof(unsigned char), result.w*result.h, in_file);
//    /* for (int i = 0; i < result.w*result.h; ++i)
//     {
//         result.img[i] = (int)temp[i];
//     }*/
//         tempw = result.w;
//         temph = result.h;
//         for (i = 0; i < 8; ++i)
//         {
//             result.w/=2;
//             result.h/=2;
//         }
//         result.h *=256;
//         result.w *=256;
//         if (result.h!=temph)
//             result.h +=256;

//         if (result.w!=tempw)
//         result.w +=256;

//         //result = newImage(result.w,result.h);
//     result.img = (int *)malloc(result.w * result.h * sizeof(int));
//         printf("converted to : %d x %d\n", result.w, result.h);
//         for (y = 0; y <result.h; ++y)
//         {
//             for (x = 0; x <result.w; ++x)
//             {
//                     if (x>=tempw || y>=temph)
//                     {
//                         result.img[x+result.w*y] = 0;
//                     }
//                     else
//                     {
//                         result.img[x+result.w*y] = (int)temp[x + tempw*y];
//                     }
//             }
            
//         }
//         /*
//         result.w = tempw;
//         result.h = temph;*/
//     fclose(in_file);
//     acl_free(temp);
//         printf("converted to : %d x %d\n", result.w, result.h);

//     return result;
// }

// void write_pgm(Image inputimg, const char * path){
//     FILE * out_file;
//     unsigned char *temp;

//     out_file = fopen(path, "wb");
//     fprintf(out_file, "P5\n");
//     int dividor=1;
//     fprintf(out_file, "%d %d\n255\n",(inputimg.w/dividor) ,(inputimg.h/dividor));
//         printf("%d %d\n255\n",(inputimg.w/dividor) ,(inputimg.h/dividor));

//     temp = (unsigned char *) acl_malloc(sizeof(unsigned char) * (inputimg.w / dividor) * (inputimg.h / dividor));

//     int x,y;
//     double temp_pxl;

//     for (y = 0; y <(inputimg.h/dividor); ++y)
//     {

//         for (x = 0; x <(inputimg.w/dividor); ++x)
//         {   
//             temp_pxl = inputimg.img[x+(inputimg.w/dividor)*y]*255;
//             if(temp_pxl>255)
//                 temp_pxl= 255;
//             if(temp_pxl<0)
//                 temp_pxl =0;
//             temp[(x + (inputimg.w/dividor)*y)]= (unsigned char)temp_pxl;
//         }
        
//     }
//     fwrite(temp,sizeof(unsigned char), (inputimg.w/dividor)*(inputimg.h/dividor), out_file);
//     acl_free(temp);
//     fclose(out_file);
// }