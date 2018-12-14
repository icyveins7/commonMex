/*
 * Version 1.0
 * Converts array of bits to string hex representation. Note that there is a bit order reverse.
 *
 * Equivalent matlab code is :
 *
 *  datastream = blanks(length(descrambled)/8*2); % this creates the empty char array for the hex strings
 *  for byte_ind = 1:length(descrambled)/8
 *     byte = descrambled((byte_ind-1)*8+1:byte_ind*8);
 *     byte = byte(8:-1:1); % reverse the bits
 *     datastream((byte_ind-1)*2+1:byte_ind*2) = dec2hex(bin2dec(num2str(byte)),2);
 *  end
 *
*/

#include "mex.h"
#include <math.h>
#include "stdio.h"
// #include <stdint.h>
// #include <time.h>
// #include <string.h>

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    // declare variables
// 	uint8_t *descrambled;
    mxUint8 *descrambled;
	int len, NUM_BYTES;
	

	// declare outputs
	// mxChar *datastream;

    /* check for proper number of arguments */
    if (nrhs!=1){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","1 Inputs required.");
    }

	descrambled = (mxUint8*)mxGetPr(prhs[0]);
	len = (int)mxGetM(prhs[0]) * (int)mxGetN(prhs[0]);
	
	NUM_BYTES = len/8;
	// printf("%i = numbytes, len input = %i, %i \n",NUM_BYTES,(int)mxGetM(prhs[0]),(int)mxGetN(prhs[0]));
	mwSize datastreamlen[2] = {1,NUM_BYTES * 2};
	
	/* create the output matrix ===== TEST WITH COMPLEX DATA*/
    // plhs[0] = mxCreateCharArray(2,datastreamlen); // lets see if it works
    /* get a pointer to the real data in the output matrix */
    // datastream = mxGetChars(plhs[0]);
	
	// create the normal c-style string to do stuff in (which we tested to work)
	char *datastream_c = (char*)mxMalloc(sizeof(char)*datastreamlen[1]);
	
	// =============/* call the computational routine */==============
	mxUint8 byte;
	for (int i = 0; i<NUM_BYTES; i++){
		
		byte = 0; // set to 0
		for (int j = 0; j<8; j++){

			if (descrambled[i*8+j] == 1){
				byte |= 1U << j;
			}
		}
		sprintf((char*)&datastream_c[i*2],"%02X",byte);
	}
	
	// printf("%s\n",(char*)datastream_c); // for debugging
	
	//copy into final matlab output
	plhs[0] = mxCreateString(datastream_c);
	
	// free the working c-style string
	mxFree(datastream_c);
	
	
}