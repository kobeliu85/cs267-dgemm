const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 40
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
*  C := C + A * B
* where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (const int lda, int M, int N, int K, double* A, double* B, double*restrict C)
{
	int Kstop = (K>>2) <<2;
	if(Kstop == K) {
  	for (int i = 0; i < M; ++i) {
    	int jlda = 0;
    	for (int j = 0; j < N; ++j) {

      	// Unroll the loop by four, FPU is four-stage pipeline
      	double cij = C[jlda+i];

      	int klda = 0;
      	for (int k = 0; k < Kstop; k+=4) {
					double cij1 = A[klda+i]          * B[jlda+k];
					double cij2 = A[klda+i+(1*klda)] * B[jlda+k+1];
					double cij3 = A[klda+i+(2*klda)] * B[jlda+k+2];
					double cij4 = A[klda+i+(3*klda)] * B[jlda+k+3];

					cij += (cij1 + cij2) + (cij3 + cij4);

      		klda += lda*4;
    		}

				for(int k=Kstop; k<K; k++) {
					cij += A[i+klda] * B[k*jlda];
					klda += lda;
				}

        C[jlda+i] = cij;
    		
    		jlda += lda;
  		}
		}
	}
	else {
		// Nice mod4, no cleanup required
  	for (int i = 0; i < M; ++i) {
    	for (int j = 0; j < N; ++j) {
      	double cij = C[i+j*lda];
  			for(int k=0; k<K; ++k) {
					cij += A[i+k*lda] * B[k+j*lda];
  			}
  			C[i+j*lda] = cij;
  		}
		}
	}
	/*
	// If it's not a nice mod4, have to cleanup after
	else {
  	for (int i = 0; i < M; ++i) {
    	int jlda = 0;
    	for (int j = 0; j < N; ++j) {

      	// Unroll the loop by four, FPU is four-stage pipeline
      	double cij = C[jlda+i];

      	int klda = 0;
      	for (int k = 0; k < Kstop; k+=4) {
					double cij1 = A[klda+i]          * B[jlda+k];
					double cij2 = A[klda+i+(1*klda)] * B[jlda+k+1];
					double cij3 = A[klda+i+(2*klda)] * B[jlda+k+2];
					double cij4 = A[klda+i+(3*klda)] * B[jlda+k+3];

					cij += (cij1 + cij2) + (cij3 + cij4);

      		klda += lda*4;
    		}

				for(int k=Kstop; k<K; k++) {
					cij += A[i+klda] * B[k*jlda];
					klda += lda;
				}

        C[jlda+i] = cij;
    		
    		jlda += lda;
  		}
		}
	}
	*/
}

static void do_block_naive (const int lda, int M, int N, int K, double* A, double* B, double*restrict C)
{
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double cij = C[i+j*lda];
  		for(int k=0; k<K; ++k) {
				cij += A[i+k*lda] * B[k+j*lda];
  		}
  		C[i+j*lda] = cij;
  	}
	}
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double*restrict C)
{
  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
				/* Correct block dimensions if block "goes off edge of" the matrix */
				int M = min (BLOCK_SIZE, lda-i);
				int N = min (BLOCK_SIZE, lda-j);
				int K = min (BLOCK_SIZE, lda-k);

				/* Perform individual block dgemm */
				do_block_naive(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
}
