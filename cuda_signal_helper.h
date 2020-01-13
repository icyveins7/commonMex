#include <npp.h>
#include <cuda_runtime.h>
#include <stdio.h>

void fftshift_32fc(Npp32fc *idata, Npp32fc *odata, int len);
void fftshift_32f(Npp32f *idata, Npp32f *odata, int len);
void fftshift_64f(Npp64f *idata, Npp64f *odata, int len);
__global__ void makeFreq_32f(int len, Npp32f fs, Npp32f *freq);
__global__ void makeFreq_64f(int len, Npp64f fs, Npp64f *freq);
__global__ void windowArray_32fc64f(Npp32fc *in, Npp64f *freq, int len, Npp32fc *out, Npp64f startFreqRange, Npp64f endFreqRange, int *windowLen);
void dsFreqDomainRepoint_32fc(Npp32fc *in, Npp32fc **outPtr, int len, int downsampleRate, int *ds_len);
__global__ void conjArray_32fc_I(Npp32fc *in, int len);