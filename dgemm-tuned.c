#include <stdlib.h>
#include <stdio.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32
#endif

#if !defined(REGISTER_BLOCK_SIZE)
#define REGISTER_BLOCK_SIZE 4
#endif

#define min(a,b) (((a)<(b))?(a):(b))

static void print_matrix(char* header, const int lda, int M, int N, double * A)
{
	return;
	printf("%s", header);
	for(int i=0; i<M; i++) {
		printf("[");
		for(int j=0; j<N; j++) {
			printf("% .2f,", A[lda*j+i]);
		}
		printf("],\n");
	}
	printf("\n");
}

/* C += A * B
 * (m x n) = (m x k) * (k x n)
 */
																 
/* Takes in column-major block A with leading dimension lda,
 * with dimensions BLOCK_SIZE * BLOCK_SIZE,
 * and column-major panel B with leading dimension ldb,
 * with dimensions BLOCK_SIZE * lda
 */
static void gebp_opt1(const int lda, const int ldb, double* A, double* B, double*restrict C)
{
	// Pack A into aP, and transpose to row-major order
	double * restrict aP = malloc(sizeof(double)*BLOCK_SIZE*BLOCK_SIZE);
	const int ldaP = BLOCK_SIZE;
	for(int i=0; i<BLOCK_SIZE; i++) {
		for(int j=0; j<BLOCK_SIZE; j++) {
			// aP like fits nicely in cache, so stride through A contiguously
			aP[j*ldaP+i] = A[i*lda + j];
		}
	}

	//printf("aP\n");
	//print_matrix(ldaP, ldaP, ldaP, aP);

	// aP and B are both packed now, do the math
	// TODO: register blocking
	
	// Temporary Caux array stores, add back to C later
	double Caux[BLOCK_SIZE];
	// iterate over columns in B and C
	for(int i=0; i<lda; i++) {
		// Zero out Caux
		for(int j=0; j<ldb; j++) {
			Caux[j] = 0;
		}
		// iterate across rows in A
		for(int j=0; j<ldb; j++) {
			// iterate down rows of A, columns of B/Caux
			for(int k=0; k<BLOCK_SIZE; k++) {
				Caux[j] += aP[j*ldaP+k] * B[i*ldb+k];
			}
		}
		// Store Caux back to proper column of C
		for(int j=0; j<ldb; j++) {
			C[i*lda+j] += Caux[j];
			Caux[j] = 0;
		}
	}

	print_matrix("A\n", lda, ldaP, ldaP, A);
	print_matrix("B\n", ldb, ldb, lda, B);
	print_matrix("C\n", lda, ldb, lda, C);
}

/**
 * A is in column-major order with leading dimension lda
 * A dimensions are lda * BLOCK_SIZE
 * B is in column-major order with leading dimension lda, K rows
 * B dimensions are BLOCK_SIZE * lda
 */
static void gepp_blk_var1(const int lda, double* A, double* B, double *restrict C)
{
	// Pack B into column-major bP with ldb = BLOCK_SIZE
	double * restrict bP = malloc(sizeof(double)*lda*BLOCK_SIZE);
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

	free(bP);
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double*restrict C)
{
	print_matrix("A\n", lda, lda, lda, A);
	print_matrix("B\n", lda, lda, lda, B);
	print_matrix("C\n", lda, lda, lda, C);
	// Block out into a panel x panel, and call gepp
	// A iterates right by columns
	if(lda % BLOCK_SIZE == 0) {
  	for (int i = 0; i < lda; i += BLOCK_SIZE) {
			// Do super-awesome gepp_block when all dims are block size
				gepp_blk_var1(lda, A + (i*lda), B + i, C);
		}
  }
	else {
		// TODO: Call something else to handle skinny remainders
		printf("Not BLOCK_SIZE (%d): %d by %d\n", BLOCK_SIZE, lda, lda);
		return;
	}
}
