#include "mex.h"
#include "stdio.h"
#include <string.h>

#include "ipp.h"

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	const mxArray *cell_element_ptr;
	char c_array[200];
	mwIndex i;
	mwSize total_num_of_cells, buflen;
	int status;
    FILE **fp;
    Ipp16s *rawdata;
	Ipp16s *rawdata_split, *rawdata_split_i;
    
    double *data, *data_i;
    int fs;
    
    /*Extract the cotents of MATLAB cell into the C array*/
	total_num_of_cells = mxGetNumberOfElements(prhs[0]);
    fs = (int)mxGetScalar(prhs[1]);
    
    /* check for proper number of arguments */
	if (nrhs!=2){
	mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","2 Inputs required.");
	}
    	
	/* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix((mwSize)(1),(mwSize)(fs*total_num_of_cells),mxCOMPLEX);
    
    /* get a pointer to the real data in the output matrix */
    data = mxGetPr(plhs[0]);
    data_i = mxGetPi(plhs[0]);

    /* the computations */
    ippInit();
    mexPrintf("This mexfile converts interleaved complex shorts (int16s) to the matlab style of split real and imaginary double arrays.\n");
    
    fp = (FILE**)mxMalloc(sizeof(FILE*)*total_num_of_cells);
    rawdata = (Ipp16s*)ippsMalloc_16s_L(total_num_of_cells*fs*2);
	for(i=0;i<total_num_of_cells;i++){
		cell_element_ptr = mxGetCell(prhs[0],i);
		buflen = mxGetN(cell_element_ptr)*sizeof(mxChar)+1;
		status = mxGetString(cell_element_ptr,c_array,buflen);
        fp[i] = fopen(c_array,"rb");
        if (fp[i]!=NULL){
			fread(&rawdata[i*fs*2],sizeof(Ipp16s),fs*2,fp[i]);
			mexPrintf("Read file %s\n", c_array);
		}
		else{
			printf("Failed to open %s!\n", c_array);
            mexErrMsgTxt("MEX CLOSED DUE TO INACCESSIBLE FILE\n");
		}
	}
	// the data is in 16sc format now, should convert to 16s split real/imag format first, instead of going to doubles, since that uses way more space
	rawdata_split = (Ipp16s*)ippsMalloc_16s_L(total_num_of_cells*fs);
	rawdata_split_i = (Ipp16s*)ippsMalloc_16s_L(total_num_of_cells*fs);
	// convert to split
	ippsCplxToReal_16sc((Ipp16sc*)rawdata,rawdata_split,rawdata_split_i,total_num_of_cells*fs);
	// go from 16s to doubles
	ippsConvert_16s64f_Sfs(rawdata_split, data, total_num_of_cells*fs, 0); // scale factor is 2^lastArg so use 2^0 = 1
	ippsConvert_16s64f_Sfs(rawdata_split_i, data_i, total_num_of_cells*fs, 0);
	
    
    for (i=0; i<total_num_of_cells; i++){fclose(fp[i]);}
    mxFree(fp);
	ippsFree(rawdata);
	ippsFree(rawdata_split); ippsFree(rawdata_split_i);
}
