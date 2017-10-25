
SRC=main.c fast_blur.c readmap.c paris_llf.c
OBJS = $(SRC:.c=.o)

NAME=llf

CC=gcc
CFLAGS= -Wall
CFLAGS+= -O3 -std=c99 -DFREEIMAGE_LIB
LFLAGS= -lm -lfreeimage 

all: $(OBJS)
	$(CC) -o $(NAME) $(OBJS) $(LFLAGS)

.c.o :
	$(CC) $<  -c $(CFLAGS) -o $@
clean:
	rm -rf ./*_accll*
	rm -rf $(OBJS) $(NAME)
	rm -rf ./*.prof
	rm -rf ./main::*
	rm -rf ~/.nv
	rm -rf ./*_acl.c
	rm -rf ./*_ocl.c
	rm -rf ./*_acl*

image:
	convert grayscale.gray grayscale.jpg
