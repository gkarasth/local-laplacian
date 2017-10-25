
SRC=main.c fast_blur.c readmap.c paris_llf.c
OBJS = $(SRC:.c=.o)

NAME=llf

CC=/opt/Centaurus/bin/acl
CFLAGS=-pg -Wall -I.
CFLAGS+= -Wno-unused-variable -Wno-unused-function -Wno-implicit-function-declaration -DFPGA_SUPPORT -DFPGA_RIFFA_SUPPORT -DCL_API_SUFFIX__VERSION_2_0= -DCL_EXT_PREFIX__VERSION_1_2_DEPRECATED= -DCL_EXT_SUFFIX__VERSION_1_2_DEPRECATED= 
LFLAGS= -lfreeimage -lriffa

all: $(OBJS)
	$(CC) -- $(LFLAGS) -o $(NAME) $(OBJS)

.c.o :
	$(CC) $< -- -c $(CFLAGS) -o $@
clean:
	rm -rf ./*_accll*
	rm -rf $(OBJS) $(NAME)
	rm -rf ./*.prof
	rm -rf ./main::*
	rm -rf ~/.nv
	rm -rf ./*_acl.c
	rm -rf ./*_ocl.c
	rm -rf ./*_acl*
#	makesimple: main.c  Image.c fast_blur.c localLaplacian.c
#	    gcc -g -std=gnu99 main.c  Image.c fast_blur.c localLaplacian.c -lm 