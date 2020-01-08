#include <stdio.h>
#include <windows.h>
#include <process.h>
#include <string.h>
#include <stdint.h>

int readFileData(char *filepath, int numHeaderBytes, void *header, void *data, char dataType, int numDataElements);
int printArrayData(char rc, char dataType, int startIdx, int numElements, void *data);
int readMultipleUSRPData(char *dirpath, int startTime, int numFiles, char dataType, void *data, int numDataElementsPerFile);