/*
 * BASED ON mexIppTone:
 * Generates a matrix of frequency shifts, using a bunch of time shift values and a bunch of frequency values.
 *
 * steervec = mexIppSteervec(td_scan_min, td_scan_step, td_num_steps, freqlist)
 * Inputs: should be fairly obvious..
 *
 * Uses IPP to quickly recreate exp(1i*2*pi*freqlist*td_scan) which is a matrix with varying frequency COLUMNS and varying time ROWS. NOTE THE CHANGE FROM THE OLD VERSION.
 *
 * Additionally, this one is meant to be CONTIGUOUS in frequency for each td_scan value! That means that pulling out indices 0 to freq_len-1 will be for the first td_scan value. This should be a better fit to be honest...
*/

#include "mex.h"
#include <math.h>
#include "stdio.h"
#include <time.h>
#include <string.h>
#include "ipp.h"
#include <windows.h>
#include <process.h>

#define NUM_THREADS 4

void ifftshift_64fc(Ipp64fc *in, Ipp64fc *out, int len){
	if (len%2 == 0){ // even
		memcpy(&out[len/2], &in[0], (len/2) * sizeof(Ipp64fc));
		memcpy(&out[0], &in[len/2], (len/2) * sizeof(Ipp64fc));
	}
	else{ // out
		memcpy(&out[len/2], &in[0], (len/2) * sizeof(Ipp64fc)); // this should round down due to integer divisions i.e. len/2 is less than half
		memcpy(&out[0], &in[len/2], (len/2 + 1) * sizeof(Ipp64fc)); // note we need the +1 here
	}
}
	

struct thread_data{
	int thread_t_ID;
	
	double *thread_freqlist;
	double thread_td_scan_min;
	double thread_td_scan_step;
	int thread_td_num_steps;
	int thread_freq_len;
	Ipp64f *thread_err;
	
	mxComplexDouble *thread_out;
};

struct thread_data thread_data_array[NUM_THREADS];

unsigned __stdcall threaded_ippsTone(void *pArgs){
	struct thread_data *inner_data;
	inner_data = (struct thread_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	
	double *freqlist = inner_data->thread_freqlist;
	double td_scan_min = inner_data->thread_td_scan_min;
	double td_scan_step = inner_data->thread_td_scan_step;
	int td_num_steps  = inner_data->thread_td_num_steps;
	int freq_len = inner_data->thread_freq_len;
	// Ipp64f *err = inner_data->thread_err;
	
	mxComplexDouble *out = inner_data->thread_out;
	
	Ipp64fc *tmp = (Ipp64fc*)ippsMalloc_64fc_L(freq_len);
	Ipp64f fs_over_L = freqlist[1] - freqlist[0]; // assumed to be generated correctly
	Ipp64f rFreq;
	Ipp64f phase;
	Ipp64f tau;
	int i;
	
	for (i = t_ID; i < td_num_steps; i = i+NUM_THREADS){
		// i have freqlist, might as well use it to calculate the phase!
		tau = td_scan_min + i * td_scan_step;
		rFreq = tau * fs_over_L;
		rFreq = rFreq - floor(rFreq); // normalize to <= 1, this also fixes negative values!
		phase = -floor((Ipp64f)freq_len/2.0) * fs_over_L * tau * IPP_2PI; // this is the phase for the starting freq at the left most bin (after fftshift), so it's negative
		while (phase < 0){phase = phase + IPP_2PI;} // for negative values we need to make it within [0, 2pi)
		while (phase >= IPP_2PI){ phase = phase - IPP_2PI;} // similarly for beyond 2pi values..
		
		ippsTone_64fc(tmp, freq_len, 1.0, rFreq, &phase, ippAlgHintNone);
		ifftshift_64fc(tmp, (Ipp64fc*)&out[i*freq_len], freq_len);
	}
	
	// err[t_ID] = td_scan_min + i * td_scan_step;
		
	//freeing
	ippsFree(tmp);
	
	_endthreadex(0);
    return 0;
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	ippInit();
    // declare variables
	double td_scan_min, td_scan_step, *freqlist;
	int td_num_steps, freq_len;

	// declare outputs
	mxComplexDouble *out;

	int t; // for loops over threads
    HANDLE ThreadList[NUM_THREADS]; // handles to threads

	
    /* check for proper number of arguments */
    if (nrhs!=4){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","4 Inputs required.");
    }

	td_scan_min = (double)mxGetScalar(prhs[0]);
	td_scan_step = (double)mxGetScalar(prhs[1]);
	td_num_steps = (int)mxGetScalar(prhs[2]);
	freqlist = (double*)mxGetDoubles(prhs[3]);
	
	freq_len = (int)mxGetM(prhs[3]) * (int)mxGetN(prhs[3]);
	
	/* create the output matrix ===== TEST WITH COMPLEX DATA*/
    plhs[0] = mxCreateDoubleMatrix(freq_len,td_num_steps,mxCOMPLEX);
    /* get a pointer to the real data in the output matrix */
    out = mxGetComplexDoubles(plhs[0]);
	
	// // for debugging
	// Ipp64f *err = (Ipp64f*)ippsMalloc_64f(NUM_THREADS);
	
	// =============/* call the computational routine */==============
	for (t = 0; t<NUM_THREADS; t++){
		// attach input arguments
		thread_data_array[t].thread_t_ID = t;
		
		thread_data_array[t].thread_freqlist = freqlist;
		thread_data_array[t].thread_td_scan_min = td_scan_min;
		thread_data_array[t].thread_td_scan_step = td_scan_step;
		thread_data_array[t].thread_td_num_steps = td_num_steps;
		thread_data_array[t].thread_freq_len = freq_len;
		
		thread_data_array[t].thread_out = out;
		// thread_data_array[t].thread_err = err;

        ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&threaded_ippsTone,(void*)&thread_data_array[t],0,NULL);
    }
    
    WaitForMultipleObjects(NUM_THREADS,ThreadList,1,INFINITE);

	// ============== CLEANUP =================
    // close threads
    printf("Closing threads...\n");
    for(t=0;t<NUM_THREADS;t++){
        CloseHandle(ThreadList[t]);
    }
    // printf("All threads closed! \n");
	// =====================================
	
	// // debug
	// for (int i=0;i<NUM_THREADS;i++){
		// if (err[i] != 0){
			// printf("values at end of thread %f\n", err[i]);
		// }
	// }
	
	// ippsFree(err);
	
	
}