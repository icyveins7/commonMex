#include "mex.h"
#include <math.h>
#include "stdio.h"
#include <time.h>
#include <string.h>
#include "ipp.h"
#include <windows.h>
#include <process.h>

#define NUM_THREADS 8

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
	
	Ipp64fc *thread_prepdt;
	double *thread_f_grad_list;
	double *thread_f0_list;
	int thread_f_grad_listlen;
	int thread_f0_listlen;
	int thread_xlen;
	double *thread_t_arr;
	double *thread_tsq_arr;
	double thread_norm_y_aligned;
	double thread_norm_x_aligned;
	
	double *thread_qf2_f_cost;
};

struct thread_data thread_data_array[NUM_THREADS];

unsigned __stdcall threaded_ippsRFDOAcost(void *pArgs){
	struct thread_data *inner_data;
	inner_data = (struct thread_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	
	Ipp64fc *prepdt = inner_data->thread_prepdt;
	double *f_grad_list = inner_data->thread_f_grad_list;
	double *f0_list = inner_data->thread_f0_list;
	int f_grad_listlen = inner_data->thread_f_grad_listlen;
	int f0_listlen = inner_data->thread_f0_listlen;
	int xlen = inner_data->thread_xlen;
	double *t_arr = inner_data->thread_t_arr;
	double *tsq_arr = inner_data->thread_tsq_arr;
	double norm_y_aligned = inner_data->thread_norm_y_aligned;
	double norm_x_aligned = inner_data->thread_norm_x_aligned;
	
	double *qf2_f_cost = inner_data->thread_qf2_f_cost;
	// ======= end of assignments

	Ipp64f *allones = ippsMalloc_64f_L(xlen);
	ippsSet_64f(1.0, allones, xlen);
	
	int i, j;
	Ipp64f *phaseval = ippsMalloc_64f_L(xlen);
	Ipp64f *phaseval1 = ippsMalloc_64f_L(xlen);
	Ipp64fc *phaseterm = ippsMalloc_64fc_L(xlen);
	Ipp64fc *pdt = ippsMalloc_64fc_L(xlen);
	Ipp64f f_grad, f0;
	Ipp64fc pdtsum;
	Ipp64f divisor = norm_y_aligned * norm_y_aligned * norm_x_aligned * norm_x_aligned;
	Ipp64fc *tmpcost = ippsMalloc_64fc_L(f_grad_listlen);
	
	for (i = t_ID; i < f0_listlen; i = i+NUM_THREADS){
		for (j = 0; j < f_grad_listlen; j++){
			f_grad = f_grad_list[j];
			f0 = f0_list[i];
			
			// calculate the phase term
			ippsMulC_64f(t_arr, -f0 * IPP_2PI, phaseval, xlen);
			// ippsAddProductC_64f(tsq_arr, f_grad * IPP_PI, phaseval, xlen); // FMA function (R2017 ipp doesnt have 64f version..)
			ippsMulC_64f(tsq_arr, f_grad * IPP_PI, phaseval1, xlen); // so we have to do it separately..
			ippsAdd_64f_I(phaseval1, phaseval, xlen);
			
			ippsPolarToCart_64fc(allones, phaseval, phaseterm, xlen);
			
			// calculate the product
			ippsMul_64fc(prepdt, phaseterm, pdt, xlen);
			ippsSum_64fc(pdt, xlen, &pdtsum);
			
			// write to output
			tmpcost[j] = pdtsum;
		}
		// compute the tmp output as an array and save into the column with the normalisation
		ippsPowerSpectr_64fc(tmpcost, (Ipp64f*)&qf2_f_cost[i * f_grad_listlen], f_grad_listlen); // does abs()^2
		ippsDivC_64f_I(divisor, (Ipp64f*)&qf2_f_cost[i * f_grad_listlen], f_grad_listlen); // normalisation
	}
		
	//freeing
	ippsFree(allones);
	ippsFree(phaseval);
	ippsFree(phaseval1);
	ippsFree(phaseterm);
	ippsFree(pdt);
	ippsFree(tmpcost);
	
	_endthreadex(0);
    return 0;
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	ippInit();
    // declare variables
	double *f_grad_list, *f0_list;
	mxComplexDouble *conj_y_aligned, *x_aligned;
	double norm_y_aligned, norm_x_aligned;
	double *t_arr, *tsq_arr;
	
	int f_grad_listlen, f0_listlen, xlen;
	
	// declare outputs
	double *qf2_f_cost;

	int t; // for loops over threads
    HANDLE ThreadList[NUM_THREADS]; // handles to threads

	
    /* check for proper number of arguments */
    if (nrhs!=8){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","8 Inputs required.");
    }

	f_grad_list = (double*)mxGetDoubles(prhs[0]);
	f0_list = (double*)mxGetDoubles(prhs[1]);
	conj_y_aligned = (mxComplexDouble*)mxGetComplexDoubles(prhs[2]);
	x_aligned = (mxComplexDouble*)mxGetComplexDoubles(prhs[3]);
	norm_y_aligned = (double)mxGetScalar(prhs[4]);
	norm_x_aligned = (double)mxGetScalar(prhs[5]);
	t_arr = (double*)mxGetDoubles(prhs[6]);
	tsq_arr = (double*)mxGetDoubles(prhs[7]);
	
	f_grad_listlen = (int)mxGetM(prhs[0]) * (int)mxGetN(prhs[0]);
	f0_listlen = (int)mxGetM(prhs[1]) * (int)mxGetN(prhs[1]);
	xlen = (int)mxGetM(prhs[3]) * (int)mxGetN(prhs[3]);
	
	/* create the output matrix ===== TEST WITH COMPLEX DATA*/
    plhs[0] = mxCreateDoubleMatrix(f_grad_listlen,f0_listlen,mxREAL);
    /* get a pointer to the real data in the output matrix */
    qf2_f_cost = mxGetDoubles(plhs[0]);
	
	
	// === PRE-COMPUTATION ===
	Ipp64fc *prepdt = (Ipp64fc*)ippsMalloc_64fc_L(xlen);
	ippsMul_64fc((Ipp64fc*)conj_y_aligned, (Ipp64fc*)x_aligned, prepdt, xlen); // multiply the two inputs first and use this
	
	// =============/* call the computational routine */==============
	for (t = 0; t<NUM_THREADS; t++){
		// attach input arguments
		thread_data_array[t].thread_t_ID = t;
		
		thread_data_array[t].thread_prepdt = prepdt;
		thread_data_array[t].thread_f_grad_list = f_grad_list;
		thread_data_array[t].thread_f0_list = f0_list;
		thread_data_array[t].thread_f_grad_listlen = f_grad_listlen;
		thread_data_array[t].thread_f0_listlen = f0_listlen;
		thread_data_array[t].thread_xlen = xlen;
		thread_data_array[t].thread_t_arr = t_arr;
		thread_data_array[t].thread_tsq_arr = tsq_arr;
		thread_data_array[t].thread_norm_y_aligned = norm_y_aligned;
		thread_data_array[t].thread_norm_x_aligned = norm_x_aligned;

		
		thread_data_array[t].thread_qf2_f_cost = qf2_f_cost;

        ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&threaded_ippsRFDOAcost,(void*)&thread_data_array[t],0,NULL);
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
	
	ippsFree(prepdt);
	
}