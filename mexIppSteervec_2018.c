/*
 * BASED ON mexIppTone:
 * Generates a matrix of frequency shifts, using a bunch of time shift values and a bunch of frequency values.
 *
 * steervec = mexIppSteervec(td_scan_min, td_scan_step, td_num_steps, freqlist)
 * Inputs: should be fairly obvious..
 *
 * Uses IPP to quickly recreate exp(1i*2*pi*freqlist*td_scan) which is a matrix with varying frequency ROWS and varying time COLUMNS.
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

#define NUM_THREADS 4

struct thread_data{
	int thread_t_ID;
	
	double *thread_freqlist;
	double thread_td_scan_min;
	double thread_td_scan_step;
	int thread_td_num_steps;
	int thread_freq_len;
	int *thread_err;
	
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
	// int *err = inner_data->thread_err;
	
	mxComplexDouble *out = inner_data->thread_out;
	
	double fs = 1.0/td_scan_step;
	double phase = 0;
	double freq = 0;
	
	IppStatus status;
	
	for (int i = t_ID; i < freq_len; i = i+NUM_THREADS){
		freq = freqlist[i];
		// phase = 0;
		phase = IPP_2PI * freq * td_scan_min;
		while (phase < 0){ phase = phase + IPP_2PI;}
		while (phase >= IPP_2PI) { phase = phase - IPP_2PI; }
		
		if (freq>=0){

			status = ippsTone_64fc((Ipp64fc*)&out[i*td_num_steps], td_num_steps, 1.0, freq/fs, &phase, ippAlgHintNone); // ippAlgHintAccurate
		}
		else{

			status = ippsTone_64fc((Ipp64fc*)&out[i*td_num_steps], td_num_steps, 1.0, (fs+freq)/fs, &phase, ippAlgHintNone);
		}
		
		// if (status!=ippStsNoErr){
			// err[i] = 1;
		// }
		// else{
			// err[i] = 0;
		// }
	}
	
	
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
    HANDLE *ThreadList; // handles to threads
    ThreadList = (HANDLE*)mxMalloc(NUM_THREADS*sizeof(HANDLE));

	
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
    plhs[0] = mxCreateDoubleMatrix(td_num_steps,freq_len,mxCOMPLEX);
    /* get a pointer to the real data in the output matrix */
    out = mxGetComplexDoubles(plhs[0]);
	
	// // for debugging
	// Ipp32s *err = (Ipp32s*)ippsMalloc_32s(freq_len);
	
	// =============/* call the computational routine */==============
	// GROUP_AFFINITY currentGroupAffinity, newGroupAffinity;
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
        // printf("Beginning threadID %i..\n",thread_data_array[t].thread_t_ID);
		// GetThreadGroupAffinity(ThreadList[t], &currentGroupAffinity);
		// newGroupAffinity = currentGroupAffinity;
		// newGroupAffinity.Group = t%2;
		// SetThreadGroupAffinity(ThreadList[t], &newGroupAffinity, NULL);
		// GetThreadGroupAffinity(ThreadList[t], &currentGroupAffinity);
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
	// for (int i=0;i<freq_len;i++){
		// if (err[i] != 0){
			// printf("Err at freq %f\n", freqlist[i]);
		// }
	// }
	
	// ippsFree(err);
	
	
}