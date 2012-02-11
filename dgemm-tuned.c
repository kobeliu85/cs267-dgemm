#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <emmintrin.h>

const char* dgemm_desc = "Tuned blocked dgemm, based on Goto.";

// GCC with BLOCK_SIZE 64 and REG_BLOCK_SIZE 2 yields > 3GFlops
// Cray with the same settings is 1.75?

#if !defined(BLOCK_SIZE)

#define BLOCK_SIZE 64
// I've manually unrolled for 2x2 matrixes, so this has to stay 2 :/
#define REG_BLOCK_SIZE 2
#define ldaP (BLOCK_SIZE*REG_BLOCK_SIZE)
#define REG_BLOCK_ITEMS (REG_BLOCK_SIZE * REG_BLOCK_SIZE)

#endif


#define min(a,b) (((a)<(b))?(a):(b))


void print_matrix(char* header, const int lda, int M, int N, double * A)
{
	return;
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
static void gebp_opt1(const int lda, const int ldb, double*restrict A, double*restrict B, double*restrict C)
{
	// Pack A into aP, and transpose to row-major order
	static double aP[BLOCK_SIZE*BLOCK_SIZE]
		__attribute__((aligned(16)));
	// This is the number of items in one register-block-row
	//const int ldaP = BLOCK_SIZE*REG_BLOCK_SIZE;
	//const int REG_BLOCK_ITEMS = REG_BLOCK_SIZE * REG_BLOCK_SIZE;
	// Repack A for contiguous register-blocked row-major access
	// e.g. for reg-block of 2:
	//
	// 1 2 | 5 6
	// 3_4_|_7_8
	// 5 6 | 7 8
	// 5 6 | 7 8
	//
	// 1 2 3 4, 5 6 7 8, 5 6 5 6, 7 8 7 8
	
	// NOTE: Compiler couldn't vectorize this, but
	// the repacking pattern is so messy I'm not surprised.
	/*
	for(int i=0; i<BLOCK_SIZE/REG_BLOCK_SIZE; i++) {
		for(int j=0; j<BLOCK_SIZE/REG_BLOCK_SIZE; j++) {
			for(int k=0; k<REG_BLOCK_SIZE; k++) {
				for(int l=0; l<REG_BLOCK_SIZE; l++) {
					// <sub row>    <---sub column----->   <------item--------->
					aP[(i * ldaP ) + (j*REG_BLOCK_ITEMS) + (k*REG_BLOCK_SIZE)+l] = 
						A[(((j*REG_BLOCK_SIZE)+l)*lda)+(i*REG_BLOCK_SIZE)+k];
				}
			}
		}
	}
	*/
	
	// NEW REPACK: Still by submatrix, but submats are col-major not row-major
	// 1 2 | 5 6
	// 3_4_|_7_8
	// 5 6 | 7 8
	// 5 6 | 7 8
	//
	// 1 3 2 4, 5 7 6 8, 5 5 6 6, 7 7 8 8
	for(int i=0; i<BLOCK_SIZE/REG_BLOCK_SIZE; i++) {
		for(int j=0; j<BLOCK_SIZE/REG_BLOCK_SIZE; j++) {
			for(int k=0; k<REG_BLOCK_SIZE; k++) {
				for(int l=0; l<REG_BLOCK_SIZE; l++) {
					aP[(i*ldaP)+(j*REG_BLOCK_ITEMS)+(k*REG_BLOCK_SIZE)+l] = 
						A[(j*REG_BLOCK_SIZE*lda)+(i*REG_BLOCK_SIZE)+(k*lda)+l];
				}
			}
		}
	}

	//print_matrix("A\n", lda, BLOCK_SIZE, BLOCK_SIZE, A);
	//print_matrix("aP\n", REG_BLOCK_ITEMS, REG_BLOCK_ITEMS, (BLOCK_SIZE*BLOCK_SIZE) / (REG_BLOCK_ITEMS), aP);

	// aP and B are both packed now, do the math
	
	// Temporary Caux array stores, add back to C later
	// Register blocked, so REG_BLOCK_SIZE columns
	static double Caux[BLOCK_SIZE*REG_BLOCK_SIZE]
		__attribute__((aligned(16)));

	// iterate over REG_BLOCK_SIZE number columns in B and C
	for(int i=0; i<lda; i+=REG_BLOCK_SIZE) {

		// Zero out Caux
		//memset(Caux, '\0', sizeof(double)*REG_BLOCK_SIZE*BLOCK_SIZE);
		for(int j=0; j<REG_BLOCK_SIZE; j++) {
			for(int k=0; k<BLOCK_SIZE; k++) {
				Caux[j*BLOCK_SIZE+k] = 0;
			}
		}

		// iterate on reg-blocks in Caux and rows of A
		for(int j=0; j<BLOCK_SIZE/REG_BLOCK_SIZE; j++) {
			// iterate down reg-blocks column of B and across reg-block row in A
			for(int k=0; k<BLOCK_SIZE/REG_BLOCK_SIZE; k++) {

				// Load the submatrix cols from C, one in each
				// TODO: unroll the first iteration, since it starts out 0
				__m128d c0 = _mm_load_pd(&Caux[j*REG_BLOCK_SIZE]);
				__m128d c1 = _mm_load_pd(&Caux[j*REG_BLOCK_SIZE+BLOCK_SIZE]);

				// First col of A
				__m128d a0  = _mm_load_pd(&aP[(j*ldaP) + (k*REG_BLOCK_ITEMS)]);
				// Second col of A
				__m128d a1  = _mm_load_pd(&aP[(j*ldaP) + (k*REG_BLOCK_ITEMS)+REG_BLOCK_SIZE]);

				double * bTemp = &B[(k*REG_BLOCK_SIZE) + (i*BLOCK_SIZE)];

				// First col of C
				__m128d b00 = _mm_set1_pd(bTemp[0]);
				b00 = _mm_mul_pd(a0, b00);
				__m128d b10 = _mm_set1_pd(bTemp[1]);
				b10 = _mm_mul_pd(a1, b10);

				c0 = _mm_add_pd(c0, _mm_add_pd(b00, b10));
				_mm_store_pd(&Caux[j*REG_BLOCK_SIZE], c0);

				// Second col of C
				__m128d b01 = _mm_set1_pd(bTemp[0+BLOCK_SIZE]);
				b01 = _mm_mul_pd(a0, b01);
				__m128d b11 = _mm_set1_pd(bTemp[1+BLOCK_SIZE]);
				b11 = _mm_mul_pd(a1, b11);

				c1 = _mm_add_pd(c1, _mm_add_pd(b01, b11));
				_mm_store_pd(&Caux[j*REG_BLOCK_SIZE+BLOCK_SIZE], c1);

				// triple-nested loop to do matmul of two register blocks
				// Down rows of A
				for(int l=0; l<REG_BLOCK_SIZE; l++) {
					// Across columns of B, row of C
					/*
					double * restrict aTemp = 
						  &aP[(j*ldaP) + (k*REG_BLOCK_ITEMS) + (l*REG_BLOCK_SIZE)];
					double * restrict bTemp = 
						  &B[k*REG_BLOCK_SIZE+((i)*BLOCK_SIZE)];
					double * restrict cTemp = 
						&Caux[((j*REG_BLOCK_SIZE))+l]; 

						// Note that this is unrolled by 4, manually.
					cTemp[0] +=
						(aTemp[0] * bTemp[0]) + (aTemp[1] * bTemp[1]) +
						(aTemp[2] * bTemp[2]) + (aTemp[3] * bTemp[3]);
					cTemp[BLOCK_SIZE] +=
						(aTemp[0] * bTemp[BLOCK_SIZE]) + (aTemp[1] * bTemp[BLOCK_SIZE+1]) +
						(aTemp[2] * bTemp[BLOCK_SIZE+2]) + (aTemp[3] * bTemp[BLOCK_SIZE+3]);
					cTemp[2*BLOCK_SIZE] +=
						(aTemp[0] * bTemp[2*BLOCK_SIZE]) + (aTemp[1] * bTemp[2*BLOCK_SIZE+1]) +
						(aTemp[2] * bTemp[2*BLOCK_SIZE+2]) + (aTemp[3] * bTemp[2*BLOCK_SIZE+3]);
					cTemp[3*BLOCK_SIZE] +=
						(aTemp[0] * bTemp[3*BLOCK_SIZE]) + (aTemp[1] * bTemp[3*BLOCK_SIZE+1]) +
						(aTemp[2] * bTemp[3*BLOCK_SIZE+2]) + (aTemp[3] * bTemp[3*BLOCK_SIZE+3]);
						*/
					

					/*
					for(int m=0; m<REG_BLOCK_SIZE; m++) {
						// Across the row of A, down the column of B
						for(int n=0; n<REG_BLOCK_SIZE; n++) {

							Caux[((j*REG_BLOCK_SIZE))+(m*BLOCK_SIZE)+l] +=  
							//     <sub row>  <---sub column---->   <------row-------><col>
						    	aP[(j*ldaP) + (k*REG_BLOCK_ITEMS) + (l*REG_BLOCK_SIZE)+ n ] * 
              //    <----sub row---> <-----column----> <row>
						    	B[k*REG_BLOCK_SIZE+((i+m)*BLOCK_SIZE) +n];
						}
					}
					*/
				}
				// End triple-nested loop
			}
		}

		print_matrix("Caux = np.matrix([", BLOCK_SIZE, BLOCK_SIZE, REG_BLOCK_SIZE, Caux);
		
		// Store Caux back to proper column of C
		for(int j=0; j<REG_BLOCK_SIZE; j++) {
			for(int k=0; k<BLOCK_SIZE; k++) {
				C[((i+j)*lda)+k] += Caux[j*BLOCK_SIZE+k];
			}
		}
	}

	print_matrix("A = np.matrix([", lda, BLOCK_SIZE, BLOCK_SIZE, A);
	print_matrix("B = np.matrix([", ldb, ldb, lda, B);
	print_matrix("C = np.matrix([", lda, ldb, lda, C);

}

/**
 * A is in column-major order with leading dimension lda
 * A dimensions are lda * BLOCK_SIZE
 * B is in column-major order with leading dimension lda, K rows
 * B dimensions are BLOCK_SIZE * lda
 */
static void gepp_blk_var1(const int lda, double*restrict bP, double* A, double* B, double *restrict C)
{
	// Pack B into column-major bP with ldb = BLOCK_SIZE
	const int ldb = BLOCK_SIZE;
	for(int i=0; i<lda; i++) {
		for(int j=0; j<BLOCK_SIZE; j++) {
			bP[i*ldb+j] = B[i*lda+j];
		}
	}

	// Do gebp on each block in A and packed B
	// TODO: remainder case for when (lda%BLOCK_SIZE)>0
	for(int i=0; i<lda; i+=BLOCK_SIZE) {
		gebp_opt1(lda, ldb, A+i, bP, C+i);
	}

}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double*restrict C)
{
	print_matrix("A = np.matrix([", lda, lda, lda, A);
	print_matrix("B = np.matrix([", lda, lda, lda, B);
	print_matrix("C = np.matrix([", lda, lda, lda, C);
	// Block out into a panel x panel, and call gepp
	// A iterates right by columns
	
	// Reuse bP for packing throughout
	if(lda % BLOCK_SIZE == 0) {
		double * restrict bP = malloc(sizeof(double)*lda*BLOCK_SIZE);
  	for (int i = 0; i < lda; i += BLOCK_SIZE) {
			// Do super-awesome gepp_block when all dims are block size
				gepp_blk_var1(lda, bP, A + (i*lda), B + i, C);
		}
		free(bP);
  }
	else {
		// TODO: Call something else to handle skinny remainders
		//printf("Not BLOCK_SIZE (%d): %d by %d\n", BLOCK_SIZE, lda, lda);
		return;
	}
}
