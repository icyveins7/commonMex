/*
 * Performs a single fast frequency shift on data.
 *
 * tone = mexIppFreqShift(x, freq, fs)
 * Inputs: signal, frequency to shift by, sampling rate
 *
 * Attempts to functionally recreate x.*exp(1i*2*pi*freq*(0:length(data)-1)/fs) quickly.
 *
 * WARNING: The allocation and subsequent conversions to and fro interleaved complex and MATLAB's split arrays takes too long. 
 * It is better to just generate the tone and let MATLAB do the multiplication the normal way.
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


struct thread_data{
	int thread_t_ID;
	int *thread_startIdx;
	double *thread_x_r;
	double *thread_x_i;
	Ipp64fc *thread_tone;
	double *thread_out_r;
	double *thread_out_i;
	int thread_len;
};

struct thread_data thread_data_array[NUM_THREADS];

unsigned __stdcall threaded_ippMul(void *pArgs){
	struct thread_data *inner_data;
	inner_data = (struct thread_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	int *startIdx = inner_data->thread_startIdx;
	double *x_r = inner_data->thread_x_r;
	double *x_i = inner_data->thread_x_i;
	Ipp64fc *tone = inner_data->thread_tone;
	double *out_r = inner_data->thread_out_r;
	double *out_i = inner_data->thread_out_i;
	int len = inner_data->thread_len;
		
    // declarations
    Ipp64fc *temp_in, *temp_out;
	int tLength;
	
	// set length for this thread to work on
	if (t_ID == NUM_THREADS-1){ // i.e. last thread
		tLength = len - startIdx[t_ID];
	}
	else{
		tLength = startIdx[t_ID+1] - startIdx[t_ID];
	}
	
    // allocations
    temp_in = (Ipp64fc*)ippsMalloc_64fc_L(tLength);
	temp_out = (Ipp64fc*)ippsMalloc_64fc_L(tLength);

	// computations
    ippsRealToCplx_64f((Ipp64f*)&x_r[startIdx[t_ID]],(Ipp64f*)&x_i[startIdx[t_ID]], temp_in, tLength); // change to interleaved complex
	
	ippsMul_64fc(temp_in,&tone[startIdx[t_ID]],temp_out,tLength); // multiply the tone
	
	ippsCplxToReal_64fc(temp_out,(Ipp64f*)&out_r[startIdx[t_ID]],(Ipp64f*)&out_i[startIdx[t_ID]], tLength); // change back to split

	// freeing
    ippsFree(temp_in);
	ippsFree(temp_out);
	
	_endthreadex(0);
    return 0;
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	ippInit();
    // declare variables
    double *x_r, *x_i;
	double freq, fs;
	int len;
	
	// for the tone
	Ipp64fc *tone;

	// declare outputs
	double *out_r, *out_i;
	
	// //reserve stuff for threads
    int t; // for loops over threads
    HANDLE *ThreadList; // handles to threads
    ThreadList = (HANDLE*)mxMalloc(NUM_THREADS*sizeof(HANDLE));

    /* check for proper number of arguments */
    if (nrhs!=3){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","3 Inputs required.");
    }

	x_r = mxGetPr(prhs[0]); 
	x_i = mxGetPi(prhs[0]);
    freq = (double)mxGetScalar(prhs[1]);
    fs = (double)mxGetScalar(prhs[2]);
    
    len = (mwSize)(mxGetM(prhs[0]) * mxGetN(prhs[0])); // signal length
	
	/* create the output matrix ===== TEST WITH COMPLEX DATA*/
    plhs[0] = mxCreateDoubleMatrix((mwSize)1,(mwSize)(len),mxCOMPLEX);
    /* get a pointer to the real data in the output matrix */
    out_r = mxGetPr(plhs[0]); // CHECK GET PR OR GET PI BASED ON WHAT YOU ALLOCATED
    out_i = mxGetPi(plhs[0]); // THIS IS PI
	
	tone = (Ipp64fc*)ippsMalloc_64fc_L(len);
	
	// =============/* call the computational routine */==============
	int startIdx[NUM_THREADS];
	for (t=0; t<NUM_THREADS; t++){
		startIdx[t] = len/NUM_THREADS*t;
		// printf("StartIdx[%i] = %i\n", t , startIdx[t]);
	}
	
	// make the tone
	double phase = 0;
	if (freq>0){
		ippsTone_64fc(tone, len, 1.0, freq/fs, &phase, ippAlgHintAccurate);
	}
	else{
		ippsTone_64fc(tone, len, 1.0, (fs-freq)/fs, &phase, ippAlgHintAccurate);
	}
	
    //start threads
    for(t=0;t<NUM_THREADS;t++){
		// attach input arguments
		thread_data_array[t].thread_t_ID = t;
		thread_data_array[t].thread_startIdx = startIdx;
		thread_data_array[t].thread_x_r = x_r;
		thread_data_array[t].thread_x_i = x_i;
		thread_data_array[t].thread_tone = tone;
		thread_data_array[t].thread_out_r = out_r;
		thread_data_array[t].thread_out_i = out_i;
		thread_data_array[t].thread_len = len;

        ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&threaded_ippMul,(void*)&thread_data_array[t],0,NULL);
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

	ippsFree(tone);
    mxFree(ThreadList);
}