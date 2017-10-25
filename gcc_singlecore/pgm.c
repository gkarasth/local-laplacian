

void write_pgm(Image inputimg, const char * path){
    FILE * out_file;
    unsigned char *temp;

    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    int dividor=1;
    fprintf(out_file, "%d %d\n255\n",(inputimg.w/dividor) ,(inputimg.h/dividor));
        printf("%d %d\n255\n",(inputimg.w/dividor) ,(inputimg.h/dividor));

    temp = (unsigned char *) malloc(sizeof(unsigned char) * (inputimg.w / dividor) * (inputimg.h / dividor));

    int x,y;
    double temp_pxl;

    for (y = 0; y <(inputimg.h/dividor); ++y)
    {

        for (x = 0; x <(inputimg.w/dividor); ++x)
        {   
            temp_pxl = inputimg.img[x+(inputimg.w/dividor)*y]*255;
            if(temp_pxl>255)
                temp_pxl= 255;
            if(temp_pxl<0)
                temp_pxl =0;
            temp[(x + (inputimg.w/dividor)*y)]= (unsigned char)temp_pxl;
        }
        
    }
    fwrite(temp,sizeof(unsigned char), (inputimg.w/dividor)*(inputimg.h/dividor), out_file);
    free(temp);
    fclose(out_file);
}