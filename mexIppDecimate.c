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

#define NUM_THREADS 6

// static volatile int WaitForThread[NUM_THREADS]; // you might not need this any more

struct thread_data{
	int thread_t_ID;
	int *thread_startIdx;
	double *thread_x_r;
	double *thread_x_i;
	Ipp64f *thread_out_r;
	Ipp64f *thread_out_i;
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
	double *x_r = inner_data->thread_x_r;
	double *x_i = inner_data->thread_x_i;
	Ipp64f *out_r = inner_data->thread_out_r;
	Ipp64f *out_i = inner_data->thread_out_i;
	int siglen = inner_data->thread_siglen;
	int numTaps = inner_data->thread_numTaps;
	double Wn = inner_data->thread_Wn;
		
    // declarations
    Ipp64f *pTaps;
	int DlyLen;
    Ipp64f *pDlySrc_r, *pDlySrc_i;
    Ipp8u *gen_pBuffer, *SR_pBuffer;
    IppsFIRSpec_64f *pSpec;
    
    int specSize, gen_bufSize, SR_bufSize;
	int filterLength;
    
    // allocations
    pTaps = (Ipp64f*)ippsMalloc_64f_L(numTaps);
	DlyLen = numTaps-1;
	pDlySrc_r = (Ipp64f*)ippsMalloc_64f_L(DlyLen);
	pDlySrc_i = (Ipp64f*)ippsMalloc_64f_L(DlyLen);

    // start making filter taps
    ippsFIRGenGetBufferSize(numTaps, &gen_bufSize);
    gen_pBuffer = ippsMalloc_8u_L(gen_bufSize);
    ippsFIRGenLowpass_64f(Wn/2.0, pTaps, numTaps, ippWinHamming, ippTrue, gen_pBuffer); // generate the filter coefficients
    
    // make the filter 
    ippsFIRSRGetSize(numTaps, ipp64f, &specSize, &SR_bufSize);
    SR_pBuffer = ippsMalloc_8u_L(SR_bufSize);
    pSpec = (IppsFIRSpec_64f*)ippsMalloc_8u_L(specSize);
    ippsFIRSRInit_64f(pTaps, numTaps, ippAlgFFT, pSpec); // initialize filter
    
    // set filter length (the total length to use in the filter function)
	if (t_ID == NUM_THREADS-1){ // i.e. last thread
		filterLength = siglen - startIdx[t_ID];
	}
	else{
		filterLength = startIdx[t_ID+1] - startIdx[t_ID];
	}
	
	// set source delay
	if (startIdx[t_ID] >= DlyLen){
		ippsCopy_64f(&x_r[startIdx[t_ID]-DlyLen],pDlySrc_r,DlyLen);
		ippsCopy_64f(&x_i[startIdx[t_ID]-DlyLen],pDlySrc_i,DlyLen);
	}
	else{
		ippsZero_64f(pDlySrc_r, DlyLen);
		ippsZero_64f(pDlySrc_i, DlyLen);
		ippsCopy_64f(&x_r[0],&pDlySrc_r[DlyLen - startIdx[t_ID]], startIdx[t_ID]);
		ippsCopy_64f(&x_i[0],&pDlySrc_i[DlyLen - startIdx[t_ID]], startIdx[t_ID]);
	}
	
	// do the filtering
    ippsFIRSR_64f(&x_r[startIdx[t_ID]], &out_r[startIdx[t_ID]], filterLength, pSpec, pDlySrc_r, NULL, SR_pBuffer); // filter real part
	ippsFIRSR_64f(&x_i[startIdx[t_ID]], &out_i[startIdx[t_ID]], filterLength, pSpec, pDlySrc_i, NULL, SR_pBuffer); // filter imag part

    // freeing
    ippsFree(pTaps);
    ippsFree(pDlySrc_r); ippsFree(pDlySrc_i);
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
    double *x_r, *x_i;
    int numTaps, siglen, Dec;
    double Wn;
	// declare interim output
	Ipp64f *out_r, *out_i;
	// declare outputs
	double *down_r, *down_i;
	
	// //reserve stuff for threads
    int t; // for loops over threads
    HANDLE *ThreadList; // handles to threads
    ThreadList = (HANDLE*)mxMalloc(NUM_THREADS*sizeof(HANDLE));

    /* check for proper number of arguments */
    if (nrhs!=4){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","4 Inputs required.");
    }

	x_r = mxGetPr(prhs[0]); 
	x_i = mxGetPi(prhs[0]);
    numTaps = (int)mxGetScalar(prhs[1]);
    Wn = (double)mxGetScalar(prhs[2]);
    Dec = (int)mxGetScalar(prhs[3]); // decimation factor
    
    siglen = (mwSize)(mxGetM(prhs[0]) * mxGetN(prhs[0])); // signal length
	
	if (siglen%Dec!=0){
        mexErrMsgTxt("Decimation factor is not a factor of signal length!");
    }
	
	/* create the output matrix ===== TEST WITH COMPLEX DATA*/
    plhs[0] = mxCreateDoubleMatrix((mwSize)1,(mwSize)(siglen/Dec),mxCOMPLEX);
    /* get a pointer to the real data in the output matrix */
    down_r = mxGetPr(plhs[0]); // CHECK GET PR OR GET PI BASED ON WHAT YOU ALLOCATED
    down_i = mxGetPi(plhs[0]); // THIS IS PI
	
	out_r = (Ipp64f*)ippsMalloc_64f_L(siglen);
	out_i = (Ipp64f*)ippsMalloc_64f_L(siglen);
	
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
		thread_data_array[t].thread_x_r = x_r;
		thread_data_array[t].thread_x_i = x_i;
		thread_data_array[t].thread_out_r = out_r;
		thread_data_array[t].thread_out_i = out_i;
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
    ippsSampleDown_64f(out_r, siglen, down_r, &downsampledLen, Dec, &pPhase);
	ippsSampleDown_64f(out_i, siglen, down_i, &downsampledLen, Dec, &pPhase);

	ippsFree(out_r);
	ippsFree(out_i);
    mxFree(ThreadList);
}