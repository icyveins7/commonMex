/*
 * Performs WOLA FFT to channelize data.
 *
 * testwola(y,f_tap,fftlen,Dec)
 * Inputs: signal, filter tap coeffs, number of FFT bins, decimation factor
 * 
 * NOTE: THE NEW R2018A USES INTERLEAVED COMPLEX. USING THE MEX API CALLS LIKE 
 * MXGETPR, MXGETPI, WILL ALLOCATE TWICE AND THEN DISCARD THE 'SEPARATE' ARRAYS IN ORDER TO WORK PROPERLY.
 * THUS, IT USUALLY INTRODUCES MEMORY INSUFFICIENCY ERRORS/LEAKS. 
 * FIX: HERE WE CALL THE R2018A MEX API SUGGESTED FUNCTIONS. REMEMBER TO COMPILE WITH mex -R2018a.
 * DO NOT USE MWSIZE. TERRIBLE IDEA. SWAPPED EVERYTHING TO INTS. DON'T LISTEN TO MATHWORKS ADVICE.
*/

#include "mex.h"
#include <math.h>
#include "stdio.h"
#include <stdlib.h>
#include "fftw3.h"
#include <time.h>
#include <string.h>
#include <windows.h>
#include <process.h>

#define NUM_THREADS 24

// definition of thread data
struct thread_data{
	int thread_t_ID;
	
	mxComplexDouble *thread_y;
	double *thread_f_tap;
	int thread_L;
	int thread_N;
	int thread_Dec;
	int thread_nprimePts;
	
	fftw_complex *thread_fin;
	fftw_complex *thread_fout;

	mxComplexDouble *thread_out; // for R2018
};

// declare global thread stuff
struct thread_data thread_data_array[NUM_THREADS];

// test fftw_plans array on stack for threads, works
fftw_plan allplans[NUM_THREADS]; // REMEMBER TO CHECK FFTW PLANS CREATION IN THE ENTRY FUNCTION

unsigned __stdcall threaded_wola(void *pArgs){
    struct thread_data *inner_data;
	inner_data = (struct thread_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	mxComplexDouble *y = inner_data->thread_y;
	int L = inner_data->thread_L;
	int N = inner_data->thread_N;
	int Dec = inner_data->thread_Dec;
	int nprimePts = inner_data->thread_nprimePts;
	double *f_tap = inner_data->thread_f_tap;

	fftw_complex *fin = inner_data->thread_fin;
	fftw_complex *fout = inner_data->thread_fout;

	mxComplexDouble *out = inner_data->thread_out; // for R2018
	// end of assignments
    
    int nprime, n, a, b; // declare to simulate threads later
    int k;

	// pick point based on thread number

	for (nprime = t_ID; nprime<nprimePts; nprime=nprime+NUM_THREADS){
        n = nprime*Dec;
        for (a = 0; a<N; a++){
            fin[a][0] = 0; // init to 0
            fin[a][1] = 0;
            for (b = 0; b<L/N; b++){
                if (n - (b*N+a) >= 0){
                    fin[a][0] = fin[a][0] + y[n-(b*N+a)].real * f_tap[b*N+a];
                    fin[a][1] = fin[a][1] + y[n-(b*N+a)].imag * f_tap[b*N+a];
                } // fin is fftw_complex
            }
        }
        fftw_execute(allplans[t_ID]); // this should place them into another fftw_complex fout
        
		if (Dec*2 == N && nprime % 2 != 0){ // only if using overlapping channels, do some phase corrections when nprime is odd
			for (k=1; k<N; k=k+2){ //  all even k are definitely even in the product anyway
				fout[k][0] = -fout[k][0];
				fout[k][1] = -fout[k][1];
			}
		}
		
        memcpy(&out[nprime*N],fout,sizeof(mxComplexDouble)*N);
	}
	
	_endthreadex(0);
    return 0;
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    // declare variables
    int i;
    double *f_tap;
    mxComplexDouble *y;
	int nprimePts;
	int fftlen, siglen, L, Dec;
	// declare outputs
	mxComplexDouble *out;
    
	clock_t start, end;
    // for fftw stuff
    fftw_complex *fin;
    fftw_complex *fout;
	
	// //reserve stuff for threads
	// GROUP_AFFINITY currentGroupAffinity, newGroupAffinity;

    int t; // for loops over threads
    HANDLE ThreadList[NUM_THREADS]; // handles to threads
    
    /* check for proper number of arguments */
    if (nrhs!=4){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","4 Inputs required.");
    }
	
	/* check input is complex */
	if (!mxIsComplex(prhs[0])){
		mexErrMsgIdAndTxt("MATLAB:checkcomplex:inputsNotComplex","Input must be complex.");
	}

	y = mxGetComplexDoubles(prhs[0]); 
    f_tap = mxGetDoubles(prhs[1]); // no more using GetPr or GetPi
	fftlen = (int)mxGetScalar(prhs[2]); // this is the length of the fft i.e. number of channels
    Dec = (int)mxGetScalar(prhs[3]); // decimation factor
    
//     printf("y[0].re = %g, y[0].im = %g \n",y[0].real,y[0].imag);
    
    siglen = (int)(mxGetM(prhs[0]) * mxGetN(prhs[0])); // signal length
    L = (int)(mxGetM(prhs[1]) * mxGetN(prhs[1])); // no. of filter taps
	
	/* argument checks */
    if (Dec != fftlen && Dec*2 != fftlen){
        mexErrMsgTxt("PHASE CORRECTION ONLY IMPLEMENTED FOR DECIMATION = FFT LENGTH OR DECIMATION * 2 = FFT LENGTH!");
    }
	if (L%fftlen != 0){
        mexErrMsgTxt("Filter taps length must be factor multiple of fft length!");
    }
	
	nprimePts = (int)(siglen/Dec);
	
	/* create the output matrix ===== TEST WITH COMPLEX DATA*/
    plhs[0] = mxCreateDoubleMatrix(fftlen,nprimePts,mxCOMPLEX);
    /* get a pointer to the real data in the output matrix */
    out = mxGetComplexDoubles(plhs[0]);
    
    // ====== ALLOC VARS FOR FFT IN THREADS BEFORE PLANS ====================

    fin = fftw_alloc_complex(fftlen*NUM_THREADS);
    fout = fftw_alloc_complex(fftlen*NUM_THREADS);
	// ======== MAKE PLANS BEFORE COMPUTATIONS IN THREADS  ============
	start = clock();
    allplans[0] = fftw_plan_dft_1d((int)fftlen, fin, fout, FFTW_BACKWARD, FFTW_ESTIMATE); // FFTW_MEASURE seems to cut execution time by ~10%, but FFTW_ESTIMATE takes ~0.001s whereas MEASURE takes ~0.375s
    end = clock();
    printf("Time for 1st single plan measure = %f \n",(double)(end-start)/CLOCKS_PER_SEC);
    
	start = clock();
    for (i=1;i<NUM_THREADS;i++){
        allplans[i] = fftw_plan_dft_1d((int)fftlen, &fin[fftlen*i], &fout[fftlen*i], FFTW_BACKWARD, FFTW_ESTIMATE); // make the other plans, not executing them yet
    }
    end = clock();
    printf("Time for 2nd-n'th single plan measure = %f \n",(double)(end-start)/CLOCKS_PER_SEC);
	// ================================================================
	for (t=0; t<NUM_THREADS; t++){
		thread_data_array[t].thread_t_ID = t;
		
		thread_data_array[t].thread_f_tap = f_tap;
		thread_data_array[t].thread_L = L;
		thread_data_array[t].thread_N = fftlen;
		thread_data_array[t].thread_Dec = Dec;
		thread_data_array[t].thread_nprimePts = nprimePts;
		thread_data_array[t].thread_y = y;
		
		thread_data_array[t].thread_fin = &fin[t*fftlen];
		thread_data_array[t].thread_fout = &fout[t*fftlen];

		thread_data_array[t].thread_out = out; // for R2018
		
        // pthread_create(&ThreadList[t], &attr, threaded_wola, (void *)&thread_data_array[t]);
		ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&threaded_wola,(void*)&thread_data_array[t],0,NULL);

        printf("Beginning threadID %i..\n",thread_data_array[t].thread_t_ID);
	}
    
    WaitForMultipleObjects(NUM_THREADS,ThreadList,1,INFINITE);

	// ============== CLEANUP =================
    // close threads
    printf("Closing threads...\n");
    for(t=0;t<NUM_THREADS;t++){
        CloseHandle(ThreadList[t]);
//         printf("Closing threadID %i.. %i\n",(int)ThreadIDList[t],WaitForThread[t]);
    }
    printf("All threads closed! \n");

    for (i=0;i<NUM_THREADS;i++){fftw_destroy_plan(allplans[i]);}

    fftw_free(fin);
    fftw_free(fout);

}