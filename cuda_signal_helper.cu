#include <cuda_signal_helper.h>

// most basic fftshift, out of place
void fftshift_32fc(Npp32fc *idata, Npp32fc *odata, int len){
	if (len%2 == 0){ // if even
		cudaMemcpy(&odata[0], &idata[len/2], (len/2)*sizeof(Npp32fc), cudaMemcpyDeviceToDevice);
		cudaMemcpy(&odata[len/2], &idata[0], (len/2)*sizeof(Npp32fc), cudaMemcpyDeviceToDevice);
	}
	else{ // if odd
		cudaMemcpy(&odata[0], &idata[len/2], (len/2)*sizeof(Npp32fc), cudaMemcpyDeviceToDevice);
		cudaMemcpy(&odata[len/2], &idata[0], (len/2+1)*sizeof(Npp32fc), cudaMemcpyDeviceToDevice);
	}
}

// most basic fftshift, out of place
void fftshift_32f(Npp32f *idata, Npp32f *odata, int len){
	if (len%2 == 0){ // if even
		cudaMemcpy(&odata[0], &idata[len/2], (len/2)*sizeof(Npp32f), cudaMemcpyDeviceToDevice);
		cudaMemcpy(&odata[len/2], &idata[0], (len/2)*sizeof(Npp32f), cudaMemcpyDeviceToDevice);
	}
	else{ // if odd
		cudaMemcpy(&odata[0], &idata[len/2], (len/2)*sizeof(Npp32f), cudaMemcpyDeviceToDevice);
		cudaMemcpy(&odata[len/2], &idata[0], (len/2+1)*sizeof(Npp32f), cudaMemcpyDeviceToDevice);
	}
}

// most basic fftshift, out of place
void fftshift_64f(Npp64f *idata, Npp64f *odata, int len){
	if (len%2 == 0){ // if even
		cudaMemcpy(&odata[0], &idata[len/2], (len/2)*sizeof(Npp64f), cudaMemcpyDeviceToDevice);
		cudaMemcpy(&odata[len/2], &idata[0], (len/2)*sizeof(Npp64f), cudaMemcpyDeviceToDevice);
	}
	else{ // if odd
		cudaMemcpy(&odata[0], &idata[len/2], (len/2)*sizeof(Npp64f), cudaMemcpyDeviceToDevice);
		cudaMemcpy(&odata[len/2], &idata[0], (len/2+1)*sizeof(Npp64f), cudaMemcpyDeviceToDevice);
	}
}

// frequency vector creation kernel
__global__ void makeFreq_32f(int len, Npp32f fs, Npp32f *freq){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x; 
	
	for (int i = index; i < len; i = i + stride){
		freq[i] = i * fs / (Npp32f)len;
		if (freq[i] >= fs/2){
			freq[i] = freq[i] - fs;
		}
	}
}

// frequency vector creation kernel
__global__ void makeFreq_64f(int len, Npp64f fs, Npp64f *freq){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x; 
	
	for (int i = index; i < len; i = i + stride){
		freq[i] = i * fs / (Npp64f)len;
		if (freq[i] >= fs/2){
			freq[i] = freq[i] - fs;
		}
	}
}

// window freq mask creation, in and freq are assumed to be fftshifted
__global__ void windowArray_32fc64f(Npp32fc *in, Npp64f *freq, int len, Npp32fc *out, Npp64f startFreqRange, Npp64f endFreqRange, int *windowLen){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x; 
	
	for (int i = index; i < len; i = i + stride){
		if (freq[i] >= startFreqRange && freq[i] < endFreqRange){
			out[i].re = in[i].re;
			out[i].im = in[i].im;
			atomicAdd(windowLen, 1);
		}
		else{
			out[i].re = 0.0;
			out[i].im = 0.0;
		}
	}
	
}

// downsample in freq domain, in is assumed to be fftshifted, no copy
// currently assumed len/downsampleRate to be even, and len to be even
void dsFreqDomainRepoint_32fc(Npp32fc *in, Npp32fc **outPtr, int len, int downsampleRate, int *ds_len){
	int downsampleLen = len / downsampleRate;
	// printf("Out pointer originally at %p \nIn pointer originally at %p\n", *outPtr, in);
	*outPtr = (Npp32fc*)&in[len/2 - downsampleLen/2];
	// printf("Out pointer now at %p \nShould be equal to shifted in pointer at %p\n", *outPtr, &in[len/2 - downsampleLen/2]);
	*ds_len = downsampleLen;
}

// array conjugation kernel in place
__global__ void conjArray_32fc_I(Npp32fc *in, int len){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x; 
	
	for (int i = index; i < len; i = i + stride){
		in[i].im = -in[i].im;
	}
	
}