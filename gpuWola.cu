/*
 * Performs WOLA FFT to channelize data. GPU/CPU version.
 *
 * gpuWola(y,f_tap,N,Dec)
 * Inputs: signal, filter tap coeffs, number of FFT bins, decimation factor
 *
 * NOTE: For some reason, compiling GPU code requires explicit specification of 
 * CUDA libraries location with -L, even if environment variable is set.
*/

#include "mex.h"
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "fftw3.h"
#include <ipp.h>

#include <windows.h>
#include <process.h>

// #include <pthread.h>

#include <cuda_runtime.h>
#include <npp.h>

#define NUM_THREADS 12

// definition of thread data
struct thread_data{
	int thread_t_ID;
	int thread_L;
	int thread_N;
	int thread_Dec;
	int thread_nprime_total;
	Npp16sc *thread_h_rawdata;
	Npp16sc *thread_d_in;
	Npp16sc *thread_d_in_next;
	Npp64fc *thread_d_out;
	Npp64fc *thread_d_out_next;
	Npp64f *thread_d_ftap;
	Npp64fc *thread_h_out;
	cudaStream_t thread_stream;
	cudaStream_t thread_copystream;
	int *thread_nprime_startIdx;
	fftw_complex *thread_fout;
	// Ipp64f *thread_out_r;
	// Ipp64f *thread_out_i; // for pre R2018
	Ipp64fc *thread_out; // for R2018
};

// declare global thread stuff
struct thread_data thread_data_array[NUM_THREADS];
fftw_plan allplans[NUM_THREADS];

// the wola kernel doing the elementwise product
__global__
void wola_front(int N, int L, Npp16sc *d_in, Npp64fc *d_out, Npp64f *d_ftap)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x; 
	
	Npp64f re, im; // oh snap writing to stack first almost halves the kernel time lol
	
	for (int a = index; a<N; a+=stride){ // probably just launch total N threads, calculate blockDim required
		re = 0;
		im = 0;
		
		for (int b = 0; b < L/N; b++){
			re = re + (Npp64f)(d_in[L-1 - (b*N+a)].re) * d_ftap[b*N+a];
			im = im + (Npp64f)(d_in[L-1 - (b*N+a)].im) * d_ftap[b*N+a];
		}
		
		d_out[a].re = re;
		d_out[a].im = im;
	}
}

// this works but sometimes throws errors, maybe don't use it..
__global__ void device_copy_vector4_int16_kernel(Npp16sc *d_out, Npp16sc *d_in, int len){ // let's see if this is faster than memcpyAsyncs
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = idx; i < len/4; i += stride){
		reinterpret_cast<int4*>(d_out)[i] = reinterpret_cast<int4*>(d_in)[i]; // you're transferring 4 ints at a time, this is the same as 4 Npp16sc since the real and imag components are equal to 32 bits (one int)
	}
	
	// in one thread, process final elements
	int remainder = len%4;
	if (idx==len/4 && remainder!=0){
		while(remainder){
			idx = len - remainder--;
			d_out[idx] = d_in[idx];
		}
	}
} // you might want to check if this is really copying correctly

// if you don't have this the mexfile will crash after the second execution because lul
__host__ void cleanUp(){
	cudaDeviceReset();
}

unsigned __stdcall threaded_wola(void *pArgs){
// void *threaded_wola(void *pArgs){
	struct thread_data *inner_data;
	inner_data = (struct thread_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	int L = inner_data->thread_L;
	int N = inner_data->thread_N;
	int Dec = inner_data->thread_Dec;
	int nprime_total = inner_data->thread_nprime_total;
	Npp16sc *h_rawdata = inner_data->thread_h_rawdata;
	Npp16sc *d_in = inner_data->thread_d_in;
	Npp16sc *d_in_next = inner_data->thread_d_in_next;
	Npp64fc *d_out = inner_data->thread_d_out;
	Npp64fc *d_out_next = inner_data->thread_d_out_next;
	Npp64f *d_ftap = inner_data->thread_d_ftap;
	Npp64fc *h_out = inner_data->thread_h_out;
	cudaStream_t stream = inner_data->thread_stream;
	cudaStream_t copystream = inner_data->thread_copystream;
	int *nprime_startIdx = inner_data->thread_nprime_startIdx;
	fftw_complex *fout = inner_data->thread_fout;
	// Ipp64f *out_r = inner_data->thread_out_r;
	// Ipp64f *out_i = inner_data->thread_out_i; // for pre-R2018
	Ipp64fc *out = inner_data->thread_out; // for R2018
	// end of assignments
	
	cudaError_t err;
	
    int k;
	Npp64fc *d_64fc_xfer;
	Npp16sc *d_16sc_xfer; // some pointers used for swapping
	int n, nprime, nprime_end, h_idx2copy;
	int nprime_start = nprime_startIdx[t_ID];
	if (t_ID == NUM_THREADS-1){ // if it's the last thread
		nprime_end = nprime_total; // the last index
	}
	else{
		nprime_end = nprime_startIdx[t_ID+1]; // otherwise just until before the next thread
	}
	
	for (nprime = nprime_start; nprime<nprime_end; nprime++){
		n = nprime * Dec;
		h_idx2copy = n + 1; 
		// now we start the async copies, except for the last iteration
		if(nprime != nprime_end-1){
			 cudaMemcpyAsync(&d_in_next[0], &d_in[Dec], (L-Dec)*sizeof(Npp16sc), cudaMemcpyDeviceToDevice, copystream); // here we move the array 'forward'
//			device_copy_vector4_int16_kernel<<<((L-Dec)/4 + 1024 - 1)/1024,1024,0,copystream>>>(&d_in_next[0], &d_in[Dec], (L-Dec)); // this appears to be ~5% faster, not sure if correct though
			cudaMemcpyAsync(&d_in_next[L-Dec], &h_rawdata[h_idx2copy], Dec*sizeof(Npp16sc), cudaMemcpyHostToDevice, copystream); // copy the new data at the end of the array
		}
		// while the copies are happening, do the computations
		
		// COMPUTATIONS
		wola_front<<<(N/1024)+1, 1024, 0, stream>>>(N, L, d_in, d_out_next, d_ftap); // you should just use N total threads, even if it's not concurrent!
		
		err = cudaGetLastError();
		if (err != cudaSuccess){
		    printf("Error: %s\n", cudaGetErrorString(err));
		}

		cudaStreamQuery(stream); // at this point, the memcpyasync from the previous iteration may not have finished yet..
		cudaStreamSynchronize(stream); // DO NOT REMOVE THESE OR ELSE THE FFT MAY BE OPERATING ON OLD OUTPUT!
		if(nprime != nprime_start){ // the first iteration is all zeros, hasn't processed anything yet
			fftw_execute(allplans[t_ID]); // fft the previous iteration's DeviceToHost output hout to fftw_complex *fout
            
            if (Dec*2 == N && (nprime-1) % 2 != 0){ // only if using overlapping channels, do some phase corrections when nprime is odd
                for (k=1; k<N; k=k+2){ //  all even k are definitely even in the product anyway
                    fout[k][0] = -fout[k][0];
                    fout[k][1] = -fout[k][1];
                }
            }
            
            // // convert to split arrays using IPP and copy to the final output ..
			 // ippsCplxToReal_64fc((Ipp64fc*)fout, &out_r[(nprime-1)*N], &out_i[(nprime-1)*N], N); // remember to copy to the previous index..
			// for R2018, just copy to final output
			memcpy(&out[(nprime-1)*N],fout,sizeof(Ipp64fc)*N);
		}
		// END OF COMPUTATIONS
		
		d_64fc_xfer = d_out_next;
		d_out_next = d_out; // this frees up the array for the next iteration of processing in the kernel
		d_out = d_64fc_xfer; // now we can copy this out without having to worry that the kernel is going to work on the data being copied out
		cudaMemcpyAsync(h_out, d_out, N*sizeof(Npp64fc), cudaMemcpyDeviceToHost, stream); // we use the stream to copy out, copystream to copy in
		
		cudaStreamQuery(copystream);
		cudaStreamSynchronize(copystream); // make sure the early copies into the device are done first, then switch the waiting buffer
		d_16sc_xfer = d_in; // pointer swapping
		d_in = d_in_next; // now the 'current' data points to the waiting buffer
		d_in_next = d_16sc_xfer; // and finally the 'waiting' data points back to the first buffer
    }
	
	cudaStreamQuery(stream); 
	cudaStreamSynchronize(stream);
	// you need to process hout to fout one last time
	fftw_execute(allplans[t_ID]);
	
	if (Dec*2 == N && (nprime_end-1) % 2 != 0){ // only if using overlapping channels, do some phase corrections when nprime is odd
		for (k=1; k<N; k=k+2){ //  all even k are definitely even in the product anyway
			fout[k][0] = -fout[k][0];
			fout[k][1] = -fout[k][1];
		}
	}
	
	// // and also fout to the final output 
	 // ippsCplxToReal_64fc((Ipp64fc*)fout, (Ipp64f*)&out_r[(nprime_end-1)*N], (Ipp64f*)&out_i[(nprime_end-1)*N], N); // something wrong with this line ?? and in the loop..
	// or for R2018, just copy to final output
	memcpy(&out[(nprime_end-1)*N],fout,sizeof(Ipp64fc)*N);
	
	_endthreadex(0); // only in windows
    return 0;
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    ippInit();
    size_t free_byte ; // these are for gpu checks
	size_t total_byte ;
    cudaMemGetInfo( &free_byte, &total_byte );
    printf("Before alloc, GPU memory usage: free = %f MB, total = %f MB\n",(double)free_byte/1024.0/1024.0,(double)total_byte/1024.0/1024.0);
    
    // declare variables
    // Npp16s *y, *y_i; // pre R2018
	Npp16sc *y; // R2018
    Npp64f *h_ftap; // direct matlab inputs are always in doubles
	int nprimePts;
	int N, rawdataLength, L, Dec, DlyLen;
	// declare outputs
	// double *out_r, *out_i; // pre R2018
	mxComplexDouble *out; // for R2018
    
    // for fftw stuff
    fftw_complex *fout;
	
    //reserve stuff for windows threads
    int t; // for loops over threads
	HANDLE ThreadList[NUM_THREADS]; // handles to threads
    
    // // stuff for pthreads
    // pthread_t ThreadList[NUM_THREADS];
    // pthread_attr_t attr;
    // pthread_attr_init(&attr);
    // pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    
    /* check for proper number of arguments */
    if (nrhs!=4){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","4 Inputs required.");
    }

	// y = (Npp16s*)mxGetPr(prhs[0]); 
	// y_i = (Npp16s*)mxGetPi(prhs[0]);
	y = (Npp16sc*)mxGetComplexInt16s(prhs[0]);
    h_ftap = (Npp64f*)mxGetDoubles(prhs[1]); 
	N = (int)mxGetScalar(prhs[2]); // this is the length of the fft i.e. number of channels
    Dec = (int)mxGetScalar(prhs[3]); // decimation factor
    
    rawdataLength = (int)(mxGetM(prhs[0]) * mxGetN(prhs[0])); // signal length
    L = (int)(mxGetM(prhs[1]) * mxGetN(prhs[1])); // no. of filter taps
    DlyLen = L - 1;
	
	/* argument checks */
    if (Dec != N && Dec*2 != N){
        mexErrMsgTxt("PHASE CORRECTION ONLY IMPLEMENTED FOR DECIMATION = FFT LENGTH OR DECIMATION * 2 = FFT LENGTH!");
    }
	if (L%N != 0){
        mexErrMsgTxt("Filter taps length must be factor multiple of fft length!");
    }
	
	nprimePts = (int)(rawdataLength/Dec);
	
	/* create the output matrix ===== TEST WITH COMPLEX DATA*/
    // plhs[0] = mxCreateDoubleMatrix(N,nprimePts,mxCOMPLEX);
	plhs[0] = mxCreateUninitNumericMatrix(N,nprimePts,mxDOUBLE_CLASS,mxCOMPLEX);
    /* get a pointer to the real data in the output matrix */
    // out_r = mxGetPr(plhs[0]); 
    // out_i = mxGetPi(plhs[0]); // for pre R2018
	out = mxGetComplexDoubles(plhs[0]);

    // declare extra pointer for consistency
    Npp16sc *h_rawdata;
    // // we make the interleaved input here 
    // cudaMallocHost((void**)&h_rawdata, sizeof(Npp16sc)*rawdataLength);
    // ippsRealToCplx_16s((Ipp16s*)y, (Ipp16s*)y_i, (Ipp16sc*)h_rawdata, rawdataLength);
    // or else, if the input is already interleaved for R2018 then
    h_rawdata = y;

    // cuda/fftw allocations
	Npp16sc *d_in, *d_in_next;
	Npp64f *d_ftap;
	Npp64fc *d_out, *d_out_next, *h_out;
	cudaMalloc((void**)&d_ftap, sizeof(Npp64f)*L); // this is directly copying the entire ftap
	cudaMalloc((void**)&d_in, sizeof(Npp16sc)*L*NUM_THREADS); // you will need at the most the same length as ftap
	cudaMalloc((void**)&d_in_next, sizeof(Npp16sc)*L*NUM_THREADS); // we have a 2nd waiting buffer
	cudaMalloc((void**)&d_out, sizeof(Npp64fc)*N*NUM_THREADS); // the output should be the N
	cudaMalloc((void**)&d_out_next, sizeof(Npp64fc)*N*NUM_THREADS); // and a 2nd waiting buffer for output as well
	cudaMallocHost((void**)&h_out, sizeof(Npp64fc)*N*NUM_THREADS); // for transferring the output to the host
	cudaMemcpy(d_ftap,h_ftap,L*sizeof(Npp64f), cudaMemcpyHostToDevice);
	printf("Copied ftap to device \n");
	cudaMemGetInfo( &free_byte, &total_byte );
	printf("After alloc, GPU memory usage: free = %f MB, total = %f MB\n",(double)free_byte/1024.0/1024.0,(double)total_byte/1024.0/1024.0);
	fout = fftw_alloc_complex(N*NUM_THREADS); // this is fine...
	printf("i haven't crashed after allocating fout\n");
    
    // === multi-threaded approach ===
	cudaStream_t streams[NUM_THREADS];
	cudaStream_t copystreams[NUM_THREADS];
	
	int nprime_total = rawdataLength/Dec;
	int nprime_startIdx[NUM_THREADS];
	int n_start[NUM_THREADS];
	printf("i haven't crashed after allocating streams on stack and some nprime arrays\n");
    
    // we do ALL the initial copies first, so that we don't invoke invoke async copies later on in the threads
	for (t=0;t<NUM_THREADS;t++){
		nprime_startIdx[t] = nprime_total/NUM_THREADS * t;
		n_start[t] = nprime_startIdx[t] * Dec;
		if (n_start[t]<L){ // then you will be copying less than the full 500k but at the end of the destination array
			nppsZero_16sc(&d_in[t*L], L); // zero out the starting part, you'll have to run this before launching threads though? only can use 1 stream
			cudaMemcpy(&d_in[t*L + DlyLen-n_start[t]],&h_rawdata[n_start[t]],(n_start[t]+1)*sizeof(Npp16sc), cudaMemcpyHostToDevice); 
			printf("ZEROED: Initial copy of rawdata for thread %i done at n_start = %i, nprime_start = %i \n", t, n_start[t], nprime_startIdx[t]);
		}
		else{ // otherwise you copy the full 500k, but from an earlier part of the source array (to accommodate the delay for the filter)
			cudaMemcpy(&d_in[t*L],&h_rawdata[n_start[t]-DlyLen],L*sizeof(Npp16sc),cudaMemcpyHostToDevice);
			printf("Initial copy of rawdata for thread %i done at n_start = %i, nprime_start = %i \n", t, n_start[t], nprime_startIdx[t]);
		}
		// these copies should be correct
		
		cudaStreamCreate(&streams[t]);
		cudaStreamCreate(&copystreams[t]);
		printf("Streams created for thread %i \n", t);
		
		allplans[t] = fftw_plan_dft_1d(N, (fftw_complex*)&h_out[t*N], &fout[t*N], FFTW_BACKWARD, FFTW_ESTIMATE);
		printf("Creating FFTW Plan for thread %i \n", t);
		if (allplans[t] == NULL){ printf("FFTW PLAN FAILED FOR THREAD %i \n", t);} // this is also fine..
		
	}
    
    // start threads
	for (t=0; t<NUM_THREADS; t++){
		thread_data_array[t].thread_t_ID = t;
		thread_data_array[t].thread_L = L;
		thread_data_array[t].thread_N = N;
		thread_data_array[t].thread_Dec = Dec;
		thread_data_array[t].thread_nprime_total = nprime_total;
		thread_data_array[t].thread_h_rawdata = h_rawdata;
		thread_data_array[t].thread_d_in = &d_in[t*L];
		thread_data_array[t].thread_d_in_next = &d_in_next[t*L];
		thread_data_array[t].thread_d_out = &d_out[t*N];
		thread_data_array[t].thread_d_out_next = &d_out_next[t*N];
		thread_data_array[t].thread_d_ftap = d_ftap;
		thread_data_array[t].thread_h_out = &h_out[t*N];
		thread_data_array[t].thread_stream = streams[t];
		thread_data_array[t].thread_copystream = copystreams[t];
		thread_data_array[t].thread_nprime_startIdx = nprime_startIdx;
		thread_data_array[t].thread_fout = &fout[t*N];

		// thread_data_array[t].thread_out_r = (Ipp64f*)&out_r[0];
		// thread_data_array[t].thread_out_i = (Ipp64f*)&out_i[0]; // for pre R2018
		thread_data_array[t].thread_out = (Ipp64fc*)&out[0]; // for R2018
		
        // pthread_create(&ThreadList[t], &attr, threaded_wola, (void *)&thread_data_array[t]);
		ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&threaded_wola,(void*)&thread_data_array[t],0,NULL);

        printf("Beginning threadID %i..\n",thread_data_array[t].thread_t_ID);
	}
	
    // for (int i = 0; i < NUM_THREADS; i++) {
        // if(pthread_join(ThreadList[i], NULL)) { // this essentially waits for all above threads
                // fprintf(stderr, "Error joining threadn");
                // return;
        // }
    // }
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

    for (t=0;t<NUM_THREADS;t++){
		cudaStreamDestroy(streams[t]); 
		cudaStreamDestroy(copystreams[t]);
		fftw_destroy_plan(allplans[t]);
	}
    // cudaFreeHost(h_rawdata); // you only need this in older than R2018
	cudaFreeHost(h_out);
	cudaFree(d_in); cudaFree(d_in_next);
	cudaFree(d_ftap);
	cudaFree(d_out); cudaFree(d_out_next);
	fftw_free(fout);
	// do not free h_ftap! it's your input from matlab!
   
   // cudaDeviceReset();
   // mexAtExit(cleanUp);
}
