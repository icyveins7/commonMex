// Calling routine:
// offset_metric = mexTestMasksForUnknowns(totallength, burstidxlen, guardidxlen, offsetidx, to_mask);

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
	
	int thread_burstidxlen;
	int thread_guardidxlen;
	int thread_totallength;
	int *thread_offsetidx;
	int thread_numOffsetidx;
	Ipp32f *thread_period;
	float *thread_to_mask;
	
	double *thread_offset_metric;
};

struct thread_data thread_data_array[NUM_THREADS];

unsigned __stdcall threaded_calcMaskOffsetMetric(void *pArgs){
	struct thread_data *inner_data;
	inner_data = (struct thread_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	
	int burstidxlen = inner_data->thread_burstidxlen;
	int guardidxlen = inner_data->thread_guardidxlen;
	int totallength = inner_data->thread_totallength;
	int *offsetidx = inner_data->thread_offsetidx;
	int numOffsetidx = inner_data->thread_numOffsetidx;
	Ipp32f *period = inner_data->thread_period;
	float *to_mask = inner_data->thread_to_mask;
	
	double *offset_metric = inner_data->thread_offset_metric;
	// ======= end of assignments
	
	// temporary full length mask
	Ipp32f *fullmask = ippsMalloc_32f_L(totallength);
	
	int i, j, rem;
	int offset;
	int periodlen = guardidxlen + burstidxlen;
	Ipp32f *period_ptr;
	
	for (i = t_ID; i < numOffsetidx; i = i+NUM_THREADS){
		offset = offsetidx[i];
		period_ptr = &period[offset];
		
		// fill up the fullmask
		for (j=0; j<totallength-periodlen; j=j+periodlen){
			ippsCopy_32f(&period[offset], &fullmask[j], periodlen);
		}

		// for the last copy, need to check the remainder
		rem = totallength - j;
		ippsCopy_32f(&period[offset], &fullmask[j], rem);
		
		// with the mask filled, perform the dotproduct
		ippsDotProd_32f64f((Ipp32f*)to_mask, fullmask, totallength, (Ipp64f*)&offset_metric[i]);
	}
		
	//freeing
	ippsFree(fullmask);
	
	_endthreadex(0);
    return 0;
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	ippInit();
    // declare variables
	int totallength;
	int burstidxlen;
	int guardidxlen;
	int *offsetidx;
	int numOffsetidx;
	float *to_mask;
	
	// declare outputs
	double *offset_metric;

	int t; // for loops over threads
    HANDLE ThreadList[NUM_THREADS]; // handles to threads
	
	int i;

	
    /* check for proper number of arguments */
    if (nrhs!=5){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","5 Inputs required.");
    }

	totallength = (int)mxGetScalar(prhs[0]);
	burstidxlen = (int)mxGetScalar(prhs[1]);
	guardidxlen = (int)mxGetScalar(prhs[2]);
	offsetidx = (int*)mxGetInt32s(prhs[3]);
	to_mask = (float*)mxGetSingles(prhs[4]);
	
	numOffsetidx = (int)mxGetM(prhs[3]) * (int)mxGetN(prhs[3]);
	
	// conditions on offsets
	if (!mxIsInt32(prhs[3])){
		mexErrMsgTxt("ERROR: Offsets must be type int32.");
	}
	
	if (!mxIsSingle(prhs[4])){
		mexErrMsgTxt("ERROR: Array to mask must be type single (float32).");
	}
	
	for (i=0;i<numOffsetidx;i++){
		if (offsetidx[i] < 0 || offsetidx[i] >= burstidxlen + guardidxlen){
			mexErrMsgTxt("ERROR: Offsets must be in the range [0, total period-1].");
		}
	}
	
	/* create the output matrix ===== TEST WITH COMPLEX DATA*/
    plhs[0] = mxCreateDoubleMatrix(numOffsetidx,1,mxREAL);
    /* get a pointer to the real data in the output matrix */
    offset_metric = mxGetDoubles(plhs[0]);
	
	// create a single period template (twice the length)
	Ipp32f *period = ippsMalloc_32f_L((burstidxlen+guardidxlen)*2);

	printf("Initializing template ...\n");
	// create a period template with twice the length, so we can offset easily
	for (i=0;i<burstidxlen+guardidxlen;i++){
		if(i<burstidxlen){
			period[i] = 1.0;
		}
		else{
			period[i] = 0.0;
		}
	}
	ippsCopy_32f(&period[0], &period[burstidxlen+guardidxlen], burstidxlen+guardidxlen);
	
	// //debug
	// for (i=0;i<(burstidxlen+guardidxlen)*2;i++){
		// printf("%i\n",(int)period[i]);
	// }
	// for (i=0;i<totallength;i++){
		// printf("%i\n",(int)to_mask[i]);
	// }
	// for (i=0;i<numOffsetidx;i++){
		// printf("%f\n",offset_metric[i]);
	// }
	
	// =============/* call the computational routine */==============
	for (t = 0; t<NUM_THREADS; t++){
		// attach input arguments
		thread_data_array[t].thread_t_ID = t;
		
		thread_data_array[t].thread_burstidxlen = burstidxlen;
		thread_data_array[t].thread_guardidxlen = guardidxlen;
		thread_data_array[t].thread_totallength = totallength;
		thread_data_array[t].thread_offsetidx = offsetidx;
		thread_data_array[t].thread_numOffsetidx = numOffsetidx;
		thread_data_array[t].thread_period = period;
		thread_data_array[t].thread_to_mask = to_mask;
		
		thread_data_array[t].thread_offset_metric = offset_metric;

        ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&threaded_calcMaskOffsetMetric,(void*)&thread_data_array[t],0,NULL);
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
	
	// free the period template
	ippsFree(period);
	
}