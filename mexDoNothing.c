#include "mex.h"
#include <math.h>
#include "stdio.h"
#include <time.h>
#include <string.h>
#include "ipp.h"
#include "mkl.h"
#include <windows.h>
#include <process.h>

// timing functions
double PCFreq = 0.0;
__int64 CounterStart = 0;
int StartCounter()
{
    LARGE_INTEGER li;
    if(!QueryPerformanceFrequency(&li))
    printf("QueryPerformanceFrequency failed!\n");

    PCFreq = ((double)li.QuadPart)/1000.0;

    QueryPerformanceCounter(&li);
    CounterStart = li.QuadPart;
	return (int)CounterStart;
}

int GetCounter()
{
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return (int)li.QuadPart;
}

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	// == INITIALIZE TIMING ==
	int start_t = StartCounter();
	int end_t;
	start_t = GetCounter();
	
	ippInit();
	
    // declare variables
	mxComplexDouble *cutSyms;
	int len, f_len;
	double *flist;
	double fstep, BR;
	
	// declare outputs
	double *gradlist, *indicator, *phaselist;

	
	// //reserve stuff for threads
    int t; // for loops over threads
	// HANDLE ThreadList[NUM_THREADS];

    // /* check for proper number of arguments */
    // if (nrhs!=4){
        // mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","4 Inputs required.");
    // }

	// cutSyms = mxGetComplexDoubles(prhs[0]);
	// flist = mxGetDoubles(prhs[1]);
	// fstep = (double)mxGetScalar(prhs[2]);
	// BR = (double)mxGetScalar(prhs[3]);
    
    // len = (int)(mxGetM(prhs[0]) * mxGetN(prhs[0]));
	// f_len = (int)(mxGetM(prhs[1]) * mxGetN(prhs[1]));
	
	// /* create the output matrix */
    // plhs[0] = mxCreateDoubleMatrix(1,f_len,mxREAL);
	// plhs[1] = mxCreateDoubleMatrix(1,f_len,mxREAL);
	// plhs[2] = mxCreateDoubleMatrix(1,f_len,mxREAL);
    // /* get a pointer to the real data in the output matrix */
    // gradlist = mxGetDoubles(plhs[0]);
	// indicator = mxGetDoubles(plhs[1]);
	// phaselist = mxGetDoubles(plhs[2]);
	
	
	
	// // =============/* call the computational routine */==============
	// for (t = 0; t<1; t++){
		// t_data_array[t].thread_t_ID = t;
		
		// t_data_array[t].thread_cutSyms = cutSyms;
		// t_data_array[t].thread_len = len;
		// t_data_array[t].thread_f_len = f_len;
		// t_data_array[t].thread_flist = flist;
		// t_data_array[t].thread_fstep = fstep;
		// t_data_array[t].thread_BR = BR;
		
		// t_data_array[t].thread_gradlist = gradlist;
		// t_data_array[t].thread_indicator = indicator;
		// t_data_array[t].thread_phaselist = phaselist;
		
		// mklSVDdemod((void*)&t_data_array[t]); // non-threaded call
	// }
	
	end_t = GetCounter();
	printf("Time taken COMPLETE MEX FUNC = %g ms \n",(end_t - start_t)/PCFreq);
	
	// =====================================
	

}