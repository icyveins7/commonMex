/*
 * BASED ON mexIppTone:
 * Generates a matrix of frequency shifts, using a bunch of time shift values and a bunch of frequency values.
 *
 * steervec = mexIppSteervec(td_scan_min, td_scan_step, td_scan_max, freqlist)
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

#define NUM_THREADS 24


/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	ippInit();
    // declare variables
	double freq, fs;
	int len;

	// declare outputs
	mxComplexDouble *out;

    /* check for proper number of arguments */
    if (nrhs!=3){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","3 Inputs required.");
    }

	len = (int)mxGetScalar(prhs[0]);
    freq = (double)mxGetScalar(prhs[1]);
    fs = (double)mxGetScalar(prhs[2]);
	
	/* create the output matrix ===== TEST WITH COMPLEX DATA*/
    plhs[0] = mxCreateDoubleMatrix(1,len,mxCOMPLEX);
    /* get a pointer to the real data in the output matrix */
    out = mxGetComplexDoubles(plhs[0]);
	
	// =============/* call the computational routine */==============
	double phase = 0;
	double sin_phase = IPP_2PI - IPP_PI2; // i.e. 3pi/2, see ippbase.h for definitions
	
	if (freq>=0){
		ippsTone_64fc(out, len, 1.0, freq/fs, &phase, ippAlgHintAccurate);
	}
	else{
		ippsTone_64fc(out, len, 1.0, (fs+freq)/fs, &phase, ippAlgHintAccurate);
	}
	
}