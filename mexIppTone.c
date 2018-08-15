/*
 * Generates a single fast frequency shift.
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

#define NUM_THREADS 6


/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	ippInit();
    // declare variables
	double freq, fs;
	int len;
	int i;

	// declare outputs
	double *out_r, *out_i;

    /* check for proper number of arguments */
    if (nrhs!=3){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","3 Inputs required.");
    }

	len = (int)mxGetScalar(prhs[0]);
    freq = (double)mxGetScalar(prhs[1]);
    fs = (double)mxGetScalar(prhs[2]);
	
	if (fabs(freq)>=0.5*fs){
		mexErrMsgTxt("Please limit freqOffset to +/- 0.5*fs \n");
	}
	
	/* create the output matrix ===== TEST WITH COMPLEX DATA*/
    plhs[0] = mxCreateDoubleMatrix((mwSize)1,(mwSize)(len),mxCOMPLEX);
    /* get a pointer to the real data in the output matrix */
    out_r = mxGetPr(plhs[0]); // CHECK GET PR OR GET PI BASED ON WHAT YOU ALLOCATED
    out_i = mxGetPi(plhs[0]); // THIS IS PI
	
	// =============/* call the computational routine */==============
	double phase = 0;
	double sin_phase = IPP_2PI - IPP_PI2; // i.e. 3pi/2, see ippbase.h for definitions
	
	if (freq>=0){
		ippsTone_64f(out_r, len, 1.0, freq/fs, &phase, ippAlgHintAccurate);
		ippsTone_64f(out_i, len, 1.0, freq/fs, &sin_phase, ippAlgHintAccurate);
	}
	else{
		ippsTone_64f(out_r, len, 1.0, (-freq)/fs, &phase, ippAlgHintAccurate);
		ippsTone_64f(out_i, len, 1.0, (-freq)/fs, &sin_phase, ippAlgHintAccurate);
		for (i = 0; i<len; i++){
			out_i[i] = -out_i[i]; // sin is an odd function
		}
	}
	
}