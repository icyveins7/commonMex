#include <fileMgmt.h>

// return the number of elements read after numHeaderBytes
// support int16, int32, float, double
// dataType is 's', 'i', 'f', 'd'
int readFileData(char *filepath, int numHeaderBytes, void *header, void *data, char dataType, int numDataElements){
	int numElementsRead = 0;
	
	FILE *fp;
	fp = fopen(filepath,"rb");
	if (fp==NULL){
		return -1;
	}
	else{
		// read header first
		if (numHeaderBytes > 0){
			fread(header, sizeof(char), numHeaderBytes, fp);
		}
		
		// read data
		if (dataType == 's'){
			numElementsRead = fread(data, sizeof(int16_t), numDataElements, fp);
		}
		else if (dataType == 'i'){
			numElementsRead = fread(data, sizeof(int32_t), numDataElements, fp);
		}
		else if (dataType == 'f'){
			numElementsRead = fread(data, sizeof(float), numDataElements, fp);
		}
		else if (dataType == 'd'){
			numElementsRead = fread(data, sizeof(double), numDataElements, fp);
		}
		
		fclose(fp);
	}
	
	return numElementsRead;
}

// helper function to print some array data for checking
// 'r' or 'c' for real/complex (complex interleaved assumed)
// support int16, int32, float, double
// dataType is 's', 'i', 'f', 'd'
int printArrayData(char rc, char dataType, int startIdx, int numElements, void *data){
	int i;
	int16_t *shortdata;
	int32_t *intdata;
	float *floatdata;
	double *doubledata;
	
	if (rc == 'r'){ // then it's real
		if (dataType == 's'){
			// cast
			shortdata = (int16_t*)data;
			
			for (i=startIdx; i<startIdx+numElements; i++){
				printf("Element %i: %i\n", i, shortdata[i]);
			}
		}
		else if (dataType == 'i'){
			// cast
			intdata = (int32_t*)data;
			
			for (i=startIdx; i<startIdx+numElements; i++){
				printf("Element %i: %i\n", i, intdata[i]);
			}
		}
		else if (dataType == 'f'){
			// cast
			floatdata = (float*)data;
			
			for (i=startIdx; i<startIdx+numElements; i++){
				printf("Element %i: %g\n", i, floatdata[i]);
			}
		}
		else if (dataType == 'd'){
			// cast
			doubledata = (double*)data;
			
			for (i=startIdx; i<startIdx+numElements; i++){
				printf("Element %i: %g\n", i, doubledata[i]);
			}
		}
		else{
			return -1;
		}
	}
	else if (rc == 'c'){ // complex
		if (dataType == 's'){
			// cast
			shortdata = (int16_t*)data;
			
			for (i=startIdx; i<startIdx+numElements; i++){
				printf("Element %i: %i, %i\n", i, shortdata[i*2 + 0], shortdata[i*2 + 1]);
			}
		}
		else if (dataType == 'i'){
			// cast
			intdata = (int32_t*)data;
			
			for (i=startIdx; i<startIdx+numElements; i++){
				printf("Element %i: %i, %i\n", i, intdata[i*2 + 0], intdata[i*2 + 1]);
			}
		}
		else if (dataType == 'f'){
			// cast
			floatdata = (float*)data;
			
			for (i=startIdx; i<startIdx+numElements; i++){
				printf("Element %i: %g, %g\n", i, floatdata[i*2 + 0], floatdata[i*2 + 1]);
			}
		}
		else if (dataType == 'd'){
			// cast
			doubledata = (double*)data;
			
			for (i=startIdx; i<startIdx+numElements; i++){
				printf("Element %i: %g, %g\n", i, doubledata[i*2 + 0], doubledata[i*2 + 1]);
			}
		}
		else{
			return -1;
		}
	}
	else{
		return -2;
	}
	
	return 0;
}

// note that the numDataElementsPerFile is 2 * fs
int readMultipleUSRPData(char *dirpath, int startTime, int numFiles, char dataType, void *data, int numDataElementsPerFile){
	
	int16_t *shortdata;
	int32_t *intdata;
	float *floatdata;
	double *doubledata;
	
	char filepath[MAX_PATH];
	int t;
	
	for (int i = 0; i<numFiles; i++){
		t = i + startTime;
		snprintf(filepath, MAX_PATH, "%s\\%i.bin", dirpath, t);
		printf("%s\n", filepath);
		
		if (dataType == 's'){
			shortdata = (int16_t*)data; // cast so you can move appropriate number of elements
		
			if (readFileData(filepath, 0, NULL, &shortdata[i * numDataElementsPerFile], dataType, numDataElementsPerFile) < 0){
				return -1;
			}
		}
		else if (dataType == 'i'){
			intdata = (int32_t*)data; // cast so you can move appropriate number of elements
		
			if (readFileData(filepath, 0, NULL, &intdata[i * numDataElementsPerFile], dataType, numDataElementsPerFile) < 0){
				return -1;
			}
		}
		else if (dataType == 'f'){
			floatdata = (float*)data; // cast so you can move appropriate number of elements
		
			if (readFileData(filepath, 0, NULL, &floatdata[i * numDataElementsPerFile], dataType, numDataElementsPerFile) < 0){
				return -1;
			}
		}
		else if (dataType == 'd'){
			doubledata = (double*)data; // cast so you can move appropriate number of elements
		
			if (readFileData(filepath, 0, NULL, &doubledata[i * numDataElementsPerFile], dataType, numDataElementsPerFile) < 0){
				return -1;
			}
		}
		
	}
	
	
	
	return 0;
}