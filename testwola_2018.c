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

static volatile int WaitForThread[12];
const int nThreads = 12;
// test fftw_plans array on stack for threads, works
fftw_plan allplans[12]; // REMEMBER TO CHECK FFTW PLANS CREATION IN THE ENTRY FUNCTION

unsigned __stdcall myfunc(void *pArgs){
    void **Args = (void**)pArgs;
	//declare inputs
	mxComplexDouble *y;
    double *f_tap;
	int *nprimePts_ptr, nprimePts, *N_ptr, N; // rename fftlen to N
    int *Dec_ptr, Dec, *L_ptr, L;
	fftw_complex *fin, *fout;
	//declare outputs
	mxComplexDouble *out;

	//declare thread variables
	double *ThreadID;
	int t_ID;
    
    int nprime, n, a, b, offset; // declare to simulate threads later
    int k;
	
	// assignments from Args passed in
    y = (mxComplexDouble*)Args[0];
    f_tap = (double*)Args[1];
    N_ptr = (int*)Args[2];
    N = N_ptr[0]; // length of fft
    Dec_ptr = (int*)Args[3];
    Dec = Dec_ptr[0];
    L_ptr = (int*)Args[4]; 
    L = L_ptr[0]; // no. of filter taps
    nprimePts_ptr = (int*)Args[5];
    nprimePts = nprimePts_ptr[0];
    
	// fftw related
	fin = (fftw_complex*)Args[6]; // these should be initialized as size = nThreads*fftlen
	fout = (fftw_complex*)Args[7];
	// outputs
    out = (mxComplexDouble*)Args[8];
	
	// now assign threadID within function
	ThreadID = (double*)Args[9];
    t_ID = (int)ThreadID[0];
    // allow new threads to be assigned in mexFunction
    WaitForThread[t_ID]=0;

	// pick point based on thread number
    offset = t_ID * N; // used to only fill the part of the fftw_complex used for this thread

	for (nprime = t_ID; nprime<nprimePts; nprime=nprime+nThreads){
        n = nprime*Dec;
        for (a = 0; a<N; a++){
            fin[a + offset][0] = 0; // init to 0
            fin[a + offset][1] = 0;
            for (b = 0; b<L/N; b++){
                if (n - (b*N+a) >= 0){
                    fin[a + offset][0] = fin[a + offset][0] + y[n-(b*N+a)].real * f_tap[b*N+a];
                    fin[a + offset][1] = fin[a + offset][1] + y[n-(b*N+a)].imag * f_tap[b*N+a]; // wtf why can't we read y??
                } // fin is fftw_complex
            }
        }
        fftw_execute(allplans[t_ID]); // this should place them into another fftw_complex fout
        
		if (Dec*2 == N && nprime % 2 != 0){ // only if using overlapping channels, do some phase corrections when nprime is odd
			for (k=1; k<N; k=k+2){ //  all even k are definitely even in the product anyway
				fout[offset + k][0] = -fout[offset + k][0];
				fout[offset + k][1] = -fout[offset + k][1];
			}
		}
		
        memcpy(&out[nprime*N],&fout[t_ID*N],sizeof(mxComplexDouble)*N);
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
    double *ThreadIDList;
    void **ThreadArgs;
    int t, sleep; // for loops over threads
    HANDLE *ThreadList; // handles to threads
    ThreadIDList = (double*)mxMalloc(nThreads*sizeof(double));
    ThreadList = (HANDLE*)mxMalloc(nThreads*sizeof(HANDLE));
    ThreadArgs = (void**)mxMalloc(10*sizeof(void*));
    sleep = 0;
	//assign threadIDs
    for(t=0;t<nThreads;t++){ThreadIDList[t] = t;}
    
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

    fin = fftw_alloc_complex(fftlen*nThreads);
    fout = fftw_alloc_complex(fftlen*nThreads);
	// ======== MAKE PLANS BEFORE COMPUTATIONS IN THREADS  ============
	start = clock();
    allplans[0] = fftw_plan_dft_1d((int)fftlen, fin, fout, FFTW_BACKWARD, FFTW_ESTIMATE); // FFTW_MEASURE seems to cut execution time by ~10%, but FFTW_ESTIMATE takes ~0.001s whereas MEASURE takes ~0.375s
    end = clock();
    printf("Time for 1st single plan measure = %f \n",(double)(end-start)/CLOCKS_PER_SEC);
    
	start = clock();
    for (i=1;i<nThreads;i++){
        allplans[i] = fftw_plan_dft_1d((int)fftlen, &fin[fftlen*i], &fout[fftlen*i], FFTW_BACKWARD, FFTW_ESTIMATE); // make the other plans, not executing them yet
    }
    end = clock();
    printf("Time for 2nd-n'th single plan measure = %f \n",(double)(end-start)/CLOCKS_PER_SEC);
	// ================================================================
	// ======== ATTACH VARS TO ARGS =======================
    ThreadArgs[0] = (void*)y;
    ThreadArgs[1] = (void*)f_tap;
    ThreadArgs[2] = (void*)&fftlen;
    ThreadArgs[3] = (void*)&Dec;
    ThreadArgs[4] = (void*)&L;
    ThreadArgs[5] = (void*)&nprimePts;
	// fftw related
	ThreadArgs[6] = (void*)fin;
	ThreadArgs[7] = (void*)fout;
	// outputs
	ThreadArgs[8] = (void*)out;
	
	
    // =============/* call the computational routine */==============
    //start threads
    for(t=0;t<nThreads;t++){
        while (t>0 && WaitForThread[t-1]==1){sleep=sleep+1; Sleep(1); printf("Slept %i..\n",sleep);}// wait for previous threads to assign ID within function
        ThreadArgs[9] = (void*)&ThreadIDList[t]; // assign the threadID
        WaitForThread[t] = 1;
        ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&myfunc,(void*)ThreadArgs,0,NULL);
        printf("Beginning threadID %i..%i\n",(int)ThreadIDList[t],WaitForThread[t]);
    }
    
    WaitForMultipleObjects(nThreads,ThreadList,1,INFINITE);

	// ============== CLEANUP =================
    // close threads
    printf("Closing threads...\n");
    for(t=0;t<nThreads;t++){
        CloseHandle(ThreadList[t]);
//         printf("Closing threadID %i.. %i\n",(int)ThreadIDList[t],WaitForThread[t]);
    }
    printf("All threads closed! \n");

    for (i=0;i<nThreads;i++){fftw_destroy_plan(allplans[i]);}

    fftw_free(fin);
    fftw_free(fout);
   
	mxFree(ThreadIDList);
    mxFree(ThreadList);
    mxFree(ThreadArgs);

}