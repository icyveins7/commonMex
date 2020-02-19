#include "mex.h"
#include <math.h>
#include "stdio.h"
#include <time.h>
#include <string.h>

#include <windows.h>
#include <process.h>

#define NUM_THREADS 12

struct thread_data{
	int thread_t_ID;
	
	int thread_burstidxlen;
	int thread_guardidxlen;
	int thread_totallength;
	int *thread_offsetidx;
	int thread_numOffsetidx;
	mxLogical *thread_period;
	
	mxLogical *thread_masks;
};

struct thread_data thread_data_array[NUM_THREADS];

unsigned __stdcall threaded_makeMasks(void *pArgs){
	struct thread_data *inner_data;
	inner_data = (struct thread_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	
	int burstidxlen = inner_data->thread_burstidxlen;
	int guardidxlen = inner_data->thread_guardidxlen;
	int totallength = inner_data->thread_totallength;
	int *offsetidx = inner_data->thread_offsetidx;
	int numOffsetidx = inner_data->thread_numOffsetidx;
	mxLogical *period = inner_data->thread_period;
	
	mxLogical *masks = inner_data->thread_masks;
	// ======= end of assignments
	
	
	int i, j;
	int offset;
	int periodlen = guardidxlen + burstidxlen;
	for (i = t_ID; i < numOffsetidx; i = i+NUM_THREADS){
		offset = offsetidx[i];
		
		j = 0;
		while (j < totallength){
			masks[i*totallength + j] = period[(j + offset)%periodlen];
			
			j++;
		}
	}
		
	//freeing

	
	_endthreadex(0);
    return 0;
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){

    // declare variables
	int totallength;
	int burstidxlen;
	int guardidxlen;
	int *offsetidx;
	int numOffsetidx;
	
	// declare outputs
	mxLogical *masks;

	int t; // for loops over threads
    HANDLE ThreadList[NUM_THREADS]; // handles to threads
	
	int i;

	
    /* check for proper number of arguments */
    if (nrhs!=4){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","4 Inputs required.");
    }

	totallength = (int)mxGetScalar(prhs[0]);
	burstidxlen = (int)mxGetScalar(prhs[1]);
	guardidxlen = (int)mxGetScalar(prhs[2]);
	offsetidx = (int*)mxGetInt32s(prhs[3]);
	
	numOffsetidx = (int)mxGetM(prhs[3]) * (int)mxGetN(prhs[3]);
	
	// conditions on offsets
	if (!mxIsInt32(prhs[3])){
		mexErrMsgTxt("ERROR: Offsets must be type int32.");
	}
	
	for (i=0;i<numOffsetidx;i++){
		if (offsetidx[i] < 0 || offsetidx[i] >= burstidxlen + guardidxlen){
			mexErrMsgTxt("ERROR: Offsets must be in the range [0, total period-1].");
		}
	}
	
	/* create the output matrix ===== TEST WITH COMPLEX DATA*/
    plhs[0] = mxCreateLogicalMatrix(totallength,numOffsetidx);
    /* get a pointer to the real data in the output matrix */
    masks = mxGetLogicals(plhs[0]);
	
	// create a single period template
	mxArray *periodmatrix = mxCreateLogicalMatrix(burstidxlen+guardidxlen,1);
	mxLogical *period = mxGetLogicals(periodmatrix);
	for (i=0;i<burstidxlen+guardidxlen;i++){
		if(i<burstidxlen){
			period[i] = true;
		}
		else{
			period[i] = false;
		}
	}
	
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

		thread_data_array[t].thread_masks = masks;

        ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&threaded_makeMasks,(void*)&thread_data_array[t],0,NULL);
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
	mxDestroyArray(periodmatrix);
	
}