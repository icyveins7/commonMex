/*
 * Generates a fast frequency shifts.
 *
 * tone = mexIppTone(len, freq, fs)
 * Inputs: signal length, frequency to shift by, sampling rate
 *
 * Uses IPP to quickly recreate exp(1i*2*pi*freq*(0:len-1)/fs) quickly.
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

#define NUM_THREADS 2

struct t_data{
	int thread_t_ID;
	
	int thread_len;
	double *thread_freq;
	int thread_numFreq;
	double thread_fs;
	
	mxComplexDouble *thread_out;
};
struct t_data_32f{
	int thread_t_ID;
	
	int thread_len;
	float *thread_freq;
	int thread_numFreq;
	double thread_fs;
	
	mxComplexSingle *thread_out;
};

// declare global thread stuff
struct t_data t_data_array[NUM_THREADS];
struct t_data_32f t_data_array_32f[NUM_THREADS];

unsigned __stdcall threaded_ippsTone(void *pArgs){
	struct t_data *inner_data;
	inner_data = (struct t_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	
	int len = inner_data->thread_len;
	double *freq = inner_data->thread_freq;
	int numFreq = inner_data->thread_numFreq;
	double fs = inner_data->thread_fs;
	
	mxComplexDouble *out = inner_data->thread_out; // cast it here first! or else it cannot be indexed
	// end of attached variables

	int i;
	
	double phase = 0;
	// pick point based on thread number
	for (i = t_ID; i<numFreq; i=i+NUM_THREADS){
		phase = 0;
		if (freq[i]>=0){
			ippsTone_64fc((Ipp64fc*)&out[i*len], len, 1.0, freq[i]/fs, &phase, ippAlgHintAccurate);
		}
		else{
			ippsTone_64fc((Ipp64fc*)&out[i*len], len, 1.0, (fs+freq[i])/fs, &phase, ippAlgHintAccurate);
		}
	}
	
	_endthreadex(0);
    return 0;
}

unsigned __stdcall threaded_ippsTone_32f(void *pArgs){
	struct t_data_32f *inner_data;
	inner_data = (struct t_data_32f *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	
	int len = inner_data->thread_len;
	float *freq = inner_data->thread_freq;
	int numFreq = inner_data->thread_numFreq;
	double fs = inner_data->thread_fs;
	
	mxComplexSingle *out = inner_data->thread_out; // cast it here first! or else it cannot be indexed
	// end of attached variables

	int i;
	
	float phase = 0;
	// pick point based on thread number
	for (i = t_ID; i<numFreq; i=i+NUM_THREADS){
		phase = 0;
		if (freq[i]>=0){
			ippsTone_32fc((Ipp32fc*)&out[i*len], len, 1.0, freq[i]/(float)fs, &phase, ippAlgHintAccurate);
		}
		else{
			ippsTone_32fc((Ipp32fc*)&out[i*len], len, 1.0, ((float)fs+freq[i])/(float)fs, &phase, ippAlgHintAccurate);
		}
	}
	
	_endthreadex(0);
    return 0;
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	ippInit();
    // declare variables
	double *freq;
	float *freq_32f;
	double fs;
	int len;
	int numFreq;

	// declare outputs
	mxComplexDouble *out;
	mxComplexSingle *out_32f;

    /* check for proper number of arguments */
    if (nrhs!=3){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","3 Inputs required.");
    }

	len = (int)mxGetScalar(prhs[0]);
	numFreq = (int)mxGetM(prhs[1]) * (int)mxGetN(prhs[1]);
    fs = (double)mxGetScalar(prhs[2]);
	
	//start threads
    int t; 
    HANDLE ThreadList[NUM_THREADS]; // handles to threads
	
	// try to smartly determine the type required?
	if (mxIsDouble(prhs[1])){
		// printf("Type of freq values is double-precision!\n");
		freq = mxGetDoubles(prhs[1]);
		/* create the output matrix ===== TEST WITH COMPLEX DATA*/
		plhs[0] = mxCreateDoubleMatrix(len,numFreq,mxCOMPLEX);
		/* get a pointer to the real data in the output matrix */
		out = mxGetComplexDoubles(plhs[0]);
		
		for(t=0;t<NUM_THREADS;t++){
			t_data_array[t].thread_t_ID = t;
			
			t_data_array[t].thread_len = len;
			t_data_array[t].thread_freq = freq;
			t_data_array[t].thread_numFreq = numFreq;
			t_data_array[t].thread_fs = fs;
				
			t_data_array[t].thread_out = out;
			
		
			ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&threaded_ippsTone,(void*)&t_data_array[t],0,NULL);
		}
		
		WaitForMultipleObjects(NUM_THREADS,ThreadList,1,INFINITE);
		
		// ============== CLEANUP =================
		for(t=0;t<NUM_THREADS;t++){
			CloseHandle(ThreadList[t]);
		}
	}
	else{
		// printf("Type of freq values is single-precision!\n");
		freq_32f = mxGetSingles(prhs[1]);
		/* create the output matrix ===== TEST WITH COMPLEX DATA*/
		plhs[0] = mxCreateNumericMatrix(len,numFreq, mxSINGLE_CLASS,mxCOMPLEX);
		/* get a pointer to the real data in the output matrix */
		out_32f = mxGetComplexSingles(plhs[0]);
		
		for(t=0;t<NUM_THREADS;t++){
			t_data_array_32f[t].thread_t_ID = t;
			
			t_data_array_32f[t].thread_len = len;
			t_data_array_32f[t].thread_freq = freq_32f;
			t_data_array_32f[t].thread_numFreq = numFreq;
			t_data_array_32f[t].thread_fs = fs;
				
			t_data_array_32f[t].thread_out = out_32f;
			
		
			ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&threaded_ippsTone_32f,(void*)&t_data_array_32f[t],0,NULL);
		}
		
		WaitForMultipleObjects(NUM_THREADS,ThreadList,1,INFINITE);
		
		// ============== CLEANUP =================
		for(t=0;t<NUM_THREADS;t++){
			CloseHandle(ThreadList[t]);
		}
	}
	
    
	// =============/* call the computational routine */==============
	// double phase = 0;
	// double sin_phase = IPP_2PI - IPP_PI2; // i.e. 3pi/2, see ippbase.h for definitions
	
	// if (freq>=0){
		// ippsTone_64fc(out, len, 1.0, freq/fs, &phase, ippAlgHintAccurate);
	// }
	// else{
		// ippsTone_64fc(out, len, 1.0, (fs+freq)/fs, &phase, ippAlgHintAccurate);
	// }
	

}