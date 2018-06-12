/*
 * Performs filtering and downsampling on data.
 *
 * channel = mexIppDecimate(x,numTaps,Wn,Dec)
 * Inputs: signal, filter tap number, frequency cutoff, decimation factor
 *
 * New code directly implementing filter taps (fir1 equivalent), filtering, and downsampling from IPP functions.
 *
 *
*/

#include "mex.h"
#include <math.h>
#include "stdio.h"
#include <time.h>
#include <string.h>
#include "ipp.h"
#include <windows.h>
#include <process.h>

#define NUM_THREADS 24

// static volatile int WaitForThread[NUM_THREADS]; // you might not need this any more

struct thread_data{
	int thread_t_ID;
	int *thread_startIdx;
	mxComplexDouble *thread_x;
	Ipp64fc *thread_out;
	int thread_siglen;
	int thread_numTaps;
	double thread_Wn;
};

struct thread_data thread_data_array[NUM_THREADS];

unsigned __stdcall threaded_ippFilter(void *pArgs){
	struct thread_data *inner_data;
	inner_data = (struct thread_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	int *startIdx = inner_data->thread_startIdx;
	mxComplexDouble *x = inner_data->thread_x;
	Ipp64fc *out = inner_data->thread_out;
	int siglen = inner_data->thread_siglen;
	int numTaps = inner_data->thread_numTaps;
	double Wn = inner_data->thread_Wn;
		
    // declarations
    Ipp64f *pTaps;
	Ipp64fc *pTaps_c;
	int DlyLen;
    Ipp64fc *pDlySrc;
    Ipp8u *gen_pBuffer, *SR_pBuffer;
    IppsFIRSpec_64fc *pSpec;
    
    int specSize, gen_bufSize, SR_bufSize;
	int filterLength;
    
    // allocations
    pTaps = (Ipp64f*)ippsMalloc_64f_L(numTaps);
	pTaps_c = (Ipp64fc*)ippsMalloc_64fc_L(numTaps);
	DlyLen = numTaps-1;
	pDlySrc = (Ipp64fc*)ippsMalloc_64fc_L(DlyLen);

    // start making filter taps
    ippsFIRGenGetBufferSize(numTaps, &gen_bufSize);
    gen_pBuffer = ippsMalloc_8u_L(gen_bufSize);
    ippsFIRGenLowpass_64f(Wn/2.0, pTaps, numTaps, ippWinHamming, ippTrue, gen_pBuffer); // generate the filter coefficients
    
    // make the filter 
    ippsFIRSRGetSize(numTaps, ipp64fc, &specSize, &SR_bufSize);
    SR_pBuffer = ippsMalloc_8u_L(SR_bufSize);
    pSpec = (IppsFIRSpec_64fc*)ippsMalloc_8u_L(specSize);
	ippsRealToCplx_64f(pTaps,NULL,pTaps_c,numTaps); // make complex taps with imaginary zeros
    ippsFIRSRInit_64fc(pTaps_c, numTaps, ippAlgFFT, pSpec); // initialize filter
    
    // set filter length (the total length to use in the filter function)
	if (t_ID == NUM_THREADS-1){ // i.e. last thread
		filterLength = siglen - startIdx[t_ID];
	}
	else{
		filterLength = startIdx[t_ID+1] - startIdx[t_ID];
	}
	
	// set source delay
	if (startIdx[t_ID] >= DlyLen){
		ippsCopy_64fc((Ipp64fc*)&x[startIdx[t_ID]-DlyLen],pDlySrc,DlyLen);
	}
	else{
		ippsZero_64fc(pDlySrc, DlyLen);
		ippsCopy_64fc((Ipp64fc*)&x[0],&pDlySrc[DlyLen - startIdx[t_ID]], startIdx[t_ID]);
	}
	
	// do the filtering
	ippsFIRSR_64fc((Ipp64fc*)&x[startIdx[t_ID]], (Ipp64fc*)&out[startIdx[t_ID]], filterLength, pSpec, pDlySrc, NULL, SR_pBuffer);

    // freeing
    ippsFree(pTaps); ippsFree(pTaps_c);
    ippsFree(pDlySrc);
    ippsFree(pSpec);
    ippsFree(gen_pBuffer);
    ippsFree(SR_pBuffer);
	
	_endthreadex(0);
    return 0;
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	ippInit();
    // declare variables
    mxComplexDouble *x;
    int numTaps, siglen, Dec;
    double Wn;
	// declare interim output
	Ipp64fc *out;
	// declare outputs
	mxComplexDouble *down;
	
	// //reserve stuff for threads
    int t; // for loops over threads
	HANDLE ThreadList[NUM_THREADS];

    /* check for proper number of arguments */
    if (nrhs!=4){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","4 Inputs required.");
    }

	x = mxGetComplexDoubles(prhs[0]);
    numTaps = (int)mxGetScalar(prhs[1]);
    Wn = (double)mxGetScalar(prhs[2]);
    Dec = (int)mxGetScalar(prhs[3]); // decimation factor
    
    siglen = (int)(mxGetM(prhs[0]) * mxGetN(prhs[0])); // signal length
	
	if (siglen%Dec!=0){
        mexErrMsgTxt("Decimation factor is not a factor of signal length!");
    }
	
	/* create the output matrix ===== TEST WITH COMPLEX DATA*/
    plhs[0] = mxCreateDoubleMatrix(1,siglen/Dec,mxCOMPLEX);
    /* get a pointer to the real data in the output matrix */
    down = mxGetComplexDoubles(plhs[0]);
	
	out = (Ipp64fc*)ippsMalloc_64fc_L(siglen);
	
	// =============/* call the computational routine */==============
	int startIdx[NUM_THREADS];
	for (t=0; t<NUM_THREADS; t++){
		startIdx[t] = siglen/NUM_THREADS*t;
		// printf("StartIdx[%i] = %i\n", t , startIdx[t]);
	}
    //start threads
    for(t=0;t<NUM_THREADS;t++){
		// attach input arguments
		thread_data_array[t].thread_t_ID = t;
		thread_data_array[t].thread_startIdx = startIdx;
		thread_data_array[t].thread_x = x;
		thread_data_array[t].thread_out = out;
		thread_data_array[t].thread_siglen = siglen;
		thread_data_array[t].thread_numTaps = numTaps;
		thread_data_array[t].thread_Wn = Wn;

        // WaitForThread[t] = 1;
        ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&threaded_ippFilter,(void*)&thread_data_array[t],0,NULL);
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
	// =====================================
	
	// downsampling args
	printf("Downsampling.. \n");
	int downsampledLen, pPhase;
    downsampledLen = siglen/Dec;
    pPhase = 0;
	// downsample
	ippsSampleDown_64fc((Ipp64fc*)out, siglen, (Ipp64fc*)down, &downsampledLen, Dec, &pPhase);

	ippsFree(out);
}