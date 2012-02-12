# on Franklin and Hopper, we will benchmark you against Cray LibSci, the default vendor-tuned BLAS. The Cray compiler wrappers handle all the linking. If you wish to compare with other BLAS implementations, check the NERSC documentation.
# This makefile is intended for the GNU C compiler. On Franklin and Hopper, the Portland compilers are default, so you must instruct the Cray compiler wrappers to switch to GNU: type "module swap PrgEnv-pgi PrgEnv-gnu"
# Your code must compile (with GCC) with the given CFLAGS. You may experiment with the OPT variable to invoke additional compiler options.

CC = cc 

# for gcc
#OPT = -O3 -march=barcelona -msse -msse2 -msse3 -m3dnow -mfpmath=sse -fomit-frame-pointer -funroll-loops -ffast-math
#CFLAGS = -Wall -std=gnu99 $(OPT)

# for cray
OPT = -O3 -h fp3 -h list=m -h cpu=barcelona
CFLAGS = $(OPT)

# for pgi
#OPT = -fast
#CFLAGS = $(OPT)

LDFLAGS = -Wall
# librt is needed for clock_gettime
LDLIBS = -lrt

targets = benchmark-naive benchmark-blocked benchmark-blas benchmark-tuned
objects = benchmark.o dgemm-naive.o dgemm-blocked.o dgemm-blas.o dgemm-tuned.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o dgemm-naive.o 
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked : benchmark.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o dgemm-blas.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-tuned: benchmark.o dgemm-tuned.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -S -c $(CFLAGS) $<
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
