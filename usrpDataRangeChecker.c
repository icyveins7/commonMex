#include <stdio.h>
#include <stdlib.h>

#include <process.h>
#include <windows.h>
#include <tchar.h>
#define BUFSIZE MAX_PATH

int main(int argc, char* argv[]){
	short minval = 0;
	short maxval = 0;
	int numSamp;
	int i;
	char sourcefile[BUFSIZE];
	
	if (argc == 3){
		snprintf(sourcefile, BUFSIZE, "%s", argv[1]);
		printf("Opening %s \n",sourcefile);
		
		sscanf (argv[2],"%d",&numSamp);
		printf("Reading %i samples \n",numSamp);
	}
	else{
		printf("Args are (sourcefilepath) (numSamp) \n");
		return 1;
	}
	
	short *data = (short*)malloc(numSamp*sizeof(short));
	
	FILE *fp;
	fp = fopen(sourcefile,"rb");
	fread(data,sizeof(short),numSamp,fp);
	fclose(fp);
	
	for (int i = 0; i<numSamp; i++){
		if (data[i]>maxval){maxval = data[i];}
		if (data[i]<minval){minval = data[i];}
	}
	
	printf("Maxval = %hi, minval = %hi \n", maxval, minval);
	
	
	free(data);
	return 0;
	
}