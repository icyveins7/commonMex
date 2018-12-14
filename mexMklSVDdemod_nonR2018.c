// NOTE: Static linking is required to shave off ippInit time (important here!)
// Static linking for ipp requires -lippcoremt -lippsmt -lippvmmt (note the mt and the last additional library)

// Version 1.0

#include "mex.h"
#include <math.h>
#include "stdio.h"
#include <time.h>
#include <string.h>
#include "ipp.h"
#include "mkl.h"
#include <windows.h>
#include <process.h>

// // timing functions
// double PCFreq = 0.0;
// __int64 CounterStart = 0;
// int StartCounter()
// {
//     LARGE_INTEGER li;
//     if(!QueryPerformanceFrequency(&li))
//     printf("QueryPerformanceFrequency failed!\n");
// 
//     PCFreq = ((double)li.QuadPart)/1000.0;
// 
//     QueryPerformanceCounter(&li);
//     CounterStart = li.QuadPart;
// 	return (int)CounterStart;
// }
// 
// int GetCounter()
// {
//     LARGE_INTEGER li;
//     QueryPerformanceCounter(&li);
//     return (int)li.QuadPart;
// }

struct thread_data{
	int thread_t_ID;

	Ipp64fc *thread_cutSyms;
	int thread_len;
	int thread_f_len;
	double *thread_flist;
	double thread_fstep;
	double thread_BR;
	
	double *thread_gradlist;
	double *thread_indicator;
	double *thread_phaselist;
};

struct thread_data t_data_array[1];

unsigned __stdcall mklSVDdemod(void *pArgs){ // gonna write it as if threaded, even though it's not, just for future use
	struct thread_data *inner_data;
	inner_data = (struct thread_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	
	Ipp64fc *cutSyms = inner_data->thread_cutSyms;
	int len = inner_data->thread_len;
	int f_len = inner_data->thread_f_len;
	double *flist = inner_data->thread_flist;
	double fstep = inner_data->thread_fstep;
	double BR = inner_data->thread_BR;
	
	// outputs
	double *gradlist = inner_data->thread_gradlist;
	double *indicator = inner_data->thread_indicator;
	double *phaselist = inner_data->thread_phaselist;
	// end of assignments from struct
	
	
	double *castSyms; 
	Ipp64f *thing2svd = ippsMalloc_64f_L(2*2);
	int info;
	double superb[2-1];
	Ipp64f *s = ippsMalloc_64f_L(2*2);
	Ipp64f *u = ippsMalloc_64f_L(2*2);
	
	// make some working vectors
	Ipp64fc *fsteptone = ippsMalloc_64fc_L(len);
	Ipp64fc *fcurrenttone = ippsMalloc_64fc_L(len);
	Ipp64fc *shiftChan = ippsMalloc_64fc_L(len);
	
	// prepare the tones
	double phase;
	phase = 0;
	ippsTone_64fc(fsteptone, len, 1.0, fstep/BR, &phase, ippAlgHintAccurate);
	phase = 0; // phase is set after the first call! reset it!
	if (flist[0] - fstep >= 0){ ippsTone_64fc(fcurrenttone, len, 1.0, (flist[0]-fstep)/BR, &phase, ippAlgHintAccurate);}
	else{ ippsTone_64fc(fcurrenttone, len, 1.0, (BR + flist[0] - fstep)/BR, &phase, ippAlgHintAccurate);}
	
	
	// iterate over the flist
	for (int f = 0; f<f_len; f++){
		ippsMul_64fc_I(fsteptone, fcurrenttone, len);
		ippsMul_64fc((Ipp64fc*)cutSyms, fcurrenttone, shiftChan, len);
		
		castSyms = (Ipp64f*)shiftChan; // cast to double to interpret it as a long array of len*2
		
		
		// matrix multiply into 2x2
		cblas_dgemm (CblasColMajor, CblasNoTrans, CblasTrans, 2, 2, len, 1, castSyms, 2, castSyms, 2, 0, thing2svd, 2);
		// printf("Output is \n%g %g\n%g %g\n",thing2svd[0],thing2svd[1],thing2svd[2],thing2svd[3]);
		
		// svd the 2x2
		info = LAPACKE_dgesvd(LAPACK_COL_MAJOR,'A','N', 2, 2, thing2svd, 2, s, u, 2, NULL, 2, superb);

		
		// save into outputs
		gradlist[f] = u[1]/u[0];
		indicator[f] = s[1]/s[0];
		phaselist[f] = atan2(u[1],u[0]);
	}
	
	// freeing
	ippsFree(s);
	ippsFree(u);
	ippsFree(fsteptone);
	ippsFree(fcurrenttone);
	ippsFree(shiftChan);
	ippsFree(thing2svd);
	


	// _endthreadex(0);
    return 0;
}

/* The gateway function */
// inputs are: flist, cutSyms, fstep, BR
// outputs are: gradlist, indicator, phaselist
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	ippInit();
// 	// == INITIALIZE TIMING ==
// 	int start_t = StartCounter();
// 	int end_t;
// 	start_t = GetCounter();
	
    // declare variables
	// mxComplexDouble *cutSyms;
	double *cutSyms_r, *cutSyms_i;
	int len, f_len;
	double *flist;
	double fstep, BR;
	
	// declare outputs
	double *gradlist, *indicator, *phaselist;

	
	// //reserve stuff for threads
    int t; // for loops over threads
	// HANDLE ThreadList[NUM_THREADS];

    /* check for proper number of arguments */
    if (nrhs!=4){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","4 Inputs required.");
    }

	// cutSyms = mxGetComplexDoubles(prhs[0]);
	cutSyms_r = mxGetPr(prhs[0]);
	cutSyms_i = mxGetPi(prhs[0]);
	flist = (double*)mxGetPr(prhs[1]);
	fstep = (double)mxGetScalar(prhs[2]);
	BR = (double)mxGetScalar(prhs[3]);
    
    len = (int)(mxGetM(prhs[0]) * mxGetN(prhs[0]));
	f_len = (int)(mxGetM(prhs[1]) * mxGetN(prhs[1]));
	
	/* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix(1,f_len,mxREAL);
	plhs[1] = mxCreateDoubleMatrix(1,f_len,mxREAL);
	plhs[2] = mxCreateDoubleMatrix(1,f_len,mxREAL);
    /* get a pointer to the real data in the output matrix */
    gradlist = mxGetPr(plhs[0]);
	indicator = mxGetPr(plhs[1]);
	phaselist = mxGetPr(plhs[2]);
	
	
	
	// =============/* call the computational routine */==============
	Ipp64fc *cutSyms = ippsMalloc_64fc_L(len);
	ippsRealToCplx_64f((Ipp64f*)cutSyms_r, (Ipp64f*)cutSyms_i, cutSyms, len); // convert to interleaved
	
	for (t = 0; t<1; t++){
		t_data_array[t].thread_t_ID = t;
		
		t_data_array[t].thread_cutSyms = cutSyms;
		t_data_array[t].thread_len = len;
		t_data_array[t].thread_f_len = f_len;
		t_data_array[t].thread_flist = flist;
		t_data_array[t].thread_fstep = fstep;
		t_data_array[t].thread_BR = BR;
		
		t_data_array[t].thread_gradlist = gradlist;
		t_data_array[t].thread_indicator = indicator;
		t_data_array[t].thread_phaselist = phaselist;
		
		mklSVDdemod((void*)&t_data_array[t]); // non-threaded call
	}
	
// 	end_t = GetCounter();
// 	printf("Time taken COMPLETE MEX FUNC = %g ms \n",(end_t - start_t)/PCFreq);
// 	
	// =====================================
	
	ippsFree(cutSyms);

}