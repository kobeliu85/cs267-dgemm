#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
//#include <emmintrin.h>

const char* dgemm_desc = "Tuned blocked dgemm, based on Goto.";

// GCC with BLOCK_SIZE 64 and REG_BLOCK_SIZE 2 yields > 3GFlops
// Cray with the same settings is 1.75?

#if !defined(BLOCK_SIZE)

#define BLOCK_SIZE 32

#define BLOCK_SIZE_32 32
#define BLOCK_SIZE_64 64
#define BLOCK_SIZE_128 128

// Register blocking
#define REG_BLOCK_SIZE 16
#define REG_BLOCK_ITEMS (REG_BLOCK_SIZE * REG_BLOCK_SIZE)

#endif


#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))


void print_matrix(char* header, const int lda, int M, int N, double * A)
{
	printf("\n%s\n", header);
	for(int i=0; i<M; i++) {
		printf("[");
		for(int j=0; j<N; j++) {
			printf("% .4f,", A[lda*j+i]);
		}
		printf("],\n");
	}
	printf("])\n\n");
}

/* C += A * B
 * (m x n) = (m x k) * (k x n)
 */
																 
/* Takes in column-major block A with leading dimension lda,
 * with dimensions BLOCK_SIZE * BLOCK_SIZE,
 * and column-major panel B with leading dimension ldb,
 * with dimensions BLOCK_SIZE * lda
 */
static void gebp_opt1(const int lda, const int ldb, double* A, double*restrict B, double*restrict C)
{
	const int ldaP = BLOCK_SIZE * REG_BLOCK_SIZE;
	// Pack A into aP, and transpose to row-major order
	static double aP[BLOCK_SIZE*BLOCK_SIZE]
		__attribute__((aligned(16)));

	// NEWEST REPACK
	//
	// 1 2 5 6
	// 3_4_7_8
	// 5 6 7 8
	// 5 6 7 8
	//
	// 1 3 2 4 5 7 6 8, 5 5 6 6 7 7 8 8
	// The idea is to completely linearize a row for register blocking (m_r)

	// Select the row
	for(int i=0; i<BLOCK_SIZE / REG_BLOCK_SIZE; i++) {
		// Select the column
		for(int j=0; j<BLOCK_SIZE; j++) {
			for(int k=0; k<REG_BLOCK_SIZE; k++) {
				aP[(ldaP*i) + (j*REG_BLOCK_SIZE) + k] = 
					A[(i*REG_BLOCK_SIZE) + (lda*j) + k];
			}
		}
	}

	//print_matrix("A\n", lda, BLOCK_SIZE, BLOCK_SIZE, A);
	//print_matrix("aP\n", BLOCK_SIZE * REG_BLOCK_SIZE, BLOCK_SIZE* REG_BLOCK_SIZE, BLOCK_SIZE / REG_BLOCK_SIZE, aP);

	// aP and B are both packed now, do the math
	
	// Temporary Caux array stores, add back to C later
	// Register blocked, so REG_BLOCK_SIZE columns
	static double Caux[BLOCK_SIZE*REG_BLOCK_SIZE]
		__attribute__((aligned(16)));

	// iterate over REG_BLOCK_SIZE number columns in B and C
	for(int i=0; i<lda; i+=REG_BLOCK_SIZE) {

		// Zero out Caux
		// Trust memset, it's really efficient.
		memset(Caux, '\0', sizeof(double)*REG_BLOCK_SIZE*BLOCK_SIZE);

		// Do a dumb loop and hope the cray vectorizes it for me :/

		// iterate on reg-block rows in Caux and flattened panel-rows of A
		for(int j=0; j<BLOCK_SIZE/REG_BLOCK_SIZE; j++) {
			// iterate on cols of B, cols of C
			for(int k=0; k<REG_BLOCK_SIZE; k++) {
				// iterate down the col of B, across the row of A
				double * cTemp = Caux + (j*REG_BLOCK_SIZE) + (BLOCK_SIZE*k);
				for(int l=0; l<BLOCK_SIZE; l++) {
					double bItem = B[(i*BLOCK_SIZE) + (k*BLOCK_SIZE) + l];
					// iterate on row in A (linear due to repack)
					for(int m=0; m<REG_BLOCK_SIZE; m++) {
						cTemp[m] += 
							aP[(j*ldaP) + (l*REG_BLOCK_SIZE) + m] * 
							bItem;
					}
				}
			}
		}
		
		//print_matrix("Caux = np.matrix([", BLOCK_SIZE, BLOCK_SIZE, REG_BLOCK_SIZE, Caux);
		
		// Store Caux back to proper column of C
		// Vectorized version was tested to be slightly faster
		for(int j=0; j<REG_BLOCK_SIZE; j++) {
			for(int k=0; k<BLOCK_SIZE; k++) {
				/*
				__m128d orig = _mm_loadu_pd(&C[((i+j)*lda)+k]);
				__m128d temp = _mm_load_pd(&Caux[j*BLOCK_SIZE+k]);
				temp = _mm_add_pd(temp, orig);
				_mm_storeu_pd(&C[((i+j)*lda)+k], temp);
				*/
				C[((i+j)*lda)+k] += Caux[j*BLOCK_SIZE+k];
			}
		}
	}

	//print_matrix("A = np.matrix([", lda, BLOCK_SIZE, BLOCK_SIZE, A);
	//print_matrix("B = np.matrix([", ldb, ldb, lda, B);
	//print_matrix("C = np.matrix([", lda, ldb, lda, C);

}
static void gebp_opt2(const int lda, const int ldb, double* A, double*restrict B, double*restrict C)
{
	const int ldaP = BLOCK_SIZE_64 * REG_BLOCK_SIZE;
	static double aP[BLOCK_SIZE_64*BLOCK_SIZE_64]
		__attribute__((aligned(16)));
	for(int i=0; i<BLOCK_SIZE_64 / REG_BLOCK_SIZE; i++) {
		for(int j=0; j<BLOCK_SIZE_64; j++) {
			for(int k=0; k<REG_BLOCK_SIZE; k++) {
				aP[(ldaP*i) + (j*REG_BLOCK_SIZE) + k] = 
					A[(i*REG_BLOCK_SIZE) + (lda*j) + k];
			}
		}
	}
	static double Caux[BLOCK_SIZE_64*REG_BLOCK_SIZE]
		__attribute__((aligned(16)));
	for(int i=0; i<lda; i+=REG_BLOCK_SIZE) {
		memset(Caux, '\0', sizeof(double)*REG_BLOCK_SIZE*BLOCK_SIZE_64);
		for(int j=0; j<BLOCK_SIZE_64/REG_BLOCK_SIZE; j++) {
			for(int k=0; k<REG_BLOCK_SIZE; k++) {
				double * cTemp = Caux + (j*REG_BLOCK_SIZE) + (BLOCK_SIZE_64*k);
				for(int l=0; l<BLOCK_SIZE_64; l++) {
					double bItem = B[(i*BLOCK_SIZE_64) + (k*BLOCK_SIZE_64) + l];
					for(int m=0; m<REG_BLOCK_SIZE; m++) {
						cTemp[m] += 
							aP[(j*ldaP) + (l*REG_BLOCK_SIZE) + m] * 
							bItem;
					}
				}
			}
		}
		for(int j=0; j<REG_BLOCK_SIZE; j++) {
			for(int k=0; k<BLOCK_SIZE_64; k++) {
				C[((i+j)*lda)+k] += Caux[j*BLOCK_SIZE_64+k];
			}
		}
	}
}
// 128
static void gebp_opt3(const int lda, const int ldb, double* A, double*restrict B, double*restrict C)
{
	const int ldaP = BLOCK_SIZE_128 * REG_BLOCK_SIZE;
	static double aP[BLOCK_SIZE_128*BLOCK_SIZE_128]
		__attribute__((aligned(16)));
	for(int i=0; i<BLOCK_SIZE_128 / REG_BLOCK_SIZE; i++) {
		for(int j=0; j<BLOCK_SIZE_128; j++) {
			for(int k=0; k<REG_BLOCK_SIZE; k++) {
				aP[(ldaP*i) + (j*REG_BLOCK_SIZE) + k] = 
					A[(i*REG_BLOCK_SIZE) + (lda*j) + k];
			}
		}
	}
	static double Caux[BLOCK_SIZE_128*REG_BLOCK_SIZE]
		__attribute__((aligned(16)));
	for(int i=0; i<lda; i+=REG_BLOCK_SIZE) {
		memset(Caux, '\0', sizeof(double)*REG_BLOCK_SIZE*BLOCK_SIZE_128);
		for(int j=0; j<BLOCK_SIZE_128/REG_BLOCK_SIZE; j++) {
			for(int k=0; k<REG_BLOCK_SIZE; k++) {
				double * cTemp = Caux + (j*REG_BLOCK_SIZE) + (BLOCK_SIZE_128*k);
				for(int l=0; l<BLOCK_SIZE_128; l++) {
					double bItem = B[(i*BLOCK_SIZE_128) + (k*BLOCK_SIZE_128) + l];
					for(int m=0; m<REG_BLOCK_SIZE; m++) {
						cTemp[m] += 
							aP[(j*ldaP) + (l*REG_BLOCK_SIZE) + m] * 
							bItem;
					}
				}
			}
		}
		for(int j=0; j<REG_BLOCK_SIZE; j++) {
			for(int k=0; k<BLOCK_SIZE_128; k++) {
				C[((i+j)*lda)+k] += Caux[j*BLOCK_SIZE_128+k];
			}
		}
	}
}

/**
 * A is in column-major order with leading dimension lda
 * A dimensions are lda * BLOCK_SIZE
 * B is in column-major order with leading dimension lda, K rows
 * B dimensions are BLOCK_SIZE * lda
 */

// Optimized for BLOCK_SIZE_32
static void gepp_blk_var1(const int lda, double*restrict bP, double* A, double* B, double *restrict C)
{
	// Pack B into column-major bP with ldb = BLOCK_SIZE_32
	const int ldb = BLOCK_SIZE_32;
	for(int i=0; i<lda; i++) {
		for(int j=0; j<BLOCK_SIZE_32; j++) {
			bP[i*ldb+j] = B[i*lda+j];
		}
	}

	// Do gebp on each block in A and packed B
	for(int i=0; i<lda; i+=BLOCK_SIZE_32) {
		gebp_opt1(lda, ldb, A+i, bP, C+i);
	}
}

// Optimized for BLOCK_SIZE_64
static void gepp_blk_var2(const int lda, double*restrict bP, double* A, double* B, double *restrict C)
{
	// Pack B into column-major bP with ldb = BLOCK_SIZE_64
	const int ldb = BLOCK_SIZE_64;
	for(int i=0; i<lda; i++) {
		for(int j=0; j<BLOCK_SIZE_64; j++) {
			bP[i*ldb+j] = B[i*lda+j];
		}
	}

	// Do gebp on each block in A and packed B
	for(int i=0; i<lda; i+=BLOCK_SIZE_64) {
		gebp_opt2(lda, ldb, A+i, bP, C+i);
	}
}

// Optimized for BLOCK_SIZE_128
static void gepp_blk_var3(const int lda, double*restrict bP, double* A, double* B, double *restrict C)
{
	// Pack B into column-major bP with ldb = BLOCK_SIZE_128
	const int ldb = BLOCK_SIZE_128;
	for(int i=0; i<lda; i++) {
		for(int j=0; j<BLOCK_SIZE_128; j++) {
			bP[i*ldb+j] = B[i*lda+j];
		}
	}

	// Do gebp on each block in A and packed B
	for(int i=0; i<lda; i+=BLOCK_SIZE_128) {
		gebp_opt3(lda, ldb, A+i, bP, C+i);
	}
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double*restrict C)
{
	//print_matrix("A = np.matrix([", lda, lda, lda, A);
	//print_matrix("B = np.matrix([", lda, lda, lda, B);
	//print_matrix("C = np.matrix([", lda, lda, lda, C);
	// Block out into a panel x panel, and call gepp
	// A iterates right by columns
	
	short packed = 0;
	int packed_lda = lda;
	double * restrict packed_A = A;
	double * restrict packed_B = B;
	double * restrict packed_C = C;

	
	if(lda % BLOCK_SIZE_32 != 0) {
		packed = 1;
		// Order by commonality, not like a branch here or there really matters.
		int current = 32;
		while(current < lda) {
			if(current + 32 > lda) {
				current += 32;
			}
			else if(current + 64 > lda) {
				current += 64;
			}
			else {
				current += 128;
			}
		}

		packed_lda = current;

		// This does a copy-on-write, which is somewhat dubious
		// TODO: test against an aligned + memset version
		// could also use entirely aligned loads/stores then too!
		packed_A = calloc(packed_lda*packed_lda,sizeof(double));
		packed_B = malloc(packed_lda*packed_lda*sizeof(double));
		packed_C = malloc(packed_lda*packed_lda*sizeof(double));
		// Repack it with the help of our friend memcpy
		for(int i=0; i<lda; i++) {
			memcpy(packed_A+(i*packed_lda), A + i*lda, lda * sizeof(double));
			memcpy(packed_B+(i*packed_lda), B + i*lda, lda * sizeof(double));
			memcpy(packed_C+(i*packed_lda), C + i*lda, lda * sizeof(double));
		}
	}

	// Reuse bP for packing throughout
	if(packed_lda % BLOCK_SIZE_128 == 0) {
		double * restrict bP = memalign(16, sizeof(double)*packed_lda*BLOCK_SIZE_128);
  	for (int i = 0; i < packed_lda; i += BLOCK_SIZE_128) {
				gepp_blk_var3(packed_lda, bP, packed_A + (i*packed_lda), packed_B + i, packed_C);
		}
		free(bP);
  }
	else if(packed_lda % BLOCK_SIZE_64 == 0) {
		double * restrict bP = memalign(16, sizeof(double)*packed_lda*BLOCK_SIZE_64);
  	for (int i = 0; i < packed_lda; i += BLOCK_SIZE_64) {
				gepp_blk_var2(packed_lda, bP, packed_A + (i*packed_lda), packed_B + i, packed_C);
		}
		free(bP);
  }
	else if(packed_lda % BLOCK_SIZE_32 == 0) {
		double * restrict bP = memalign(16, sizeof(double)*packed_lda*BLOCK_SIZE_32);
  	for (int i = 0; i < packed_lda; i += BLOCK_SIZE_32) {
				gepp_blk_var1(packed_lda, bP, packed_A + (i*packed_lda), packed_B + i, packed_C);
		}
		free(bP);
  }

	// Unpack the array here
  if(packed) {
  	for(int i=0; i<lda; i++) {
			memcpy(C + i*lda, packed_C+(i*packed_lda), lda * sizeof(double));
  	}
  }
}
