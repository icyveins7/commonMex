#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>

#include <process.h>
#include <windows.h>
#include <tchar.h>
#define BUFSIZE MAX_PATH
#define MAX_DIR 1024

int main(int argc, char* argv[]){
	
	TCHAR sourcedirpath[BUFSIZE];
	int sourcedirpathlen;
	TCHAR targetdirpath[BUFSIZE];
	int targetdirpathlen;
	TCHAR filename[BUFSIZE];
	TCHAR filepath[BUFSIZE];
	TCHAR targetfilepath[BUFSIZE];
	TCHAR dirfilenames[MAX_DIR][BUFSIZE];

	int storedBufferLen;
	
	if (argc == 4){

		sourcedirpathlen = snprintf(sourcedirpath, BUFSIZE, "%s", argv[1]);
		printf("Set sourcedir to %s \n",&sourcedirpath[0]);

		targetdirpathlen = snprintf(targetdirpath, BUFSIZE, "%s", argv[2]);
		printf("Set targetdir to %s \n",&targetdirpath[0]);

		sscanf (argv[3],"%d",&storedBufferLen);
		printf("Set storedBufferLen to %i \n",storedBufferLen);
	}
	else {
		printf("Arguments are (sourcedirpath) (targetdirpath) (storedBufferLen) \n");
		return 1;
	}
	
	int dirIdx;
	int countChar;
	int filenamelen;
	int filenametimes[MAX_DIR];
	int minTime;
	int minIdx;
    DIR *sourcedir, *targetdir;
	struct dirent *sourceent;
	struct dirent *targetent;
	FILE *fp;
	long fileBytes;
	while(1){
		sourcedir = opendir(sourcedirpath);
		
		dirIdx = 0;
		minTime = -1;
		while((sourceent = readdir(sourcedir))!=NULL){
			strcpy(dirfilenames[dirIdx], sourceent->d_name);
			countChar = snprintf(filepath, BUFSIZE, "%s%s", sourcedirpath, dirfilenames[dirIdx]);
			filenamelen = countChar - sourcedirpathlen;
			// printf("%s, filename length = %i \n", filepath, filenamelen);
			
			
			
			if(filenamelen > 10){
				fp = fopen(filepath, "rb");
				fseek(fp,0, SEEK_END);
				fileBytes = ftell(fp);
				fclose(fp);
				
				if (fileBytes == 200003584){
					sscanf(dirfilenames[dirIdx], "%d", &filenametimes[dirIdx]);
					// printf("Time of file = %i, size = %i bytes \n", filenametimes[dirIdx], fileBytes);

					
					if (minTime == -1){ // initialized
						minTime = filenametimes[dirIdx];
						minIdx = dirIdx;
					}
					else{
						if (filenametimes[dirIdx]<minTime){
							minTime = filenametimes[dirIdx];
							minIdx = dirIdx;
						}
					}
							
					// printf("Last four chars of file is %s \n", &filepath[countChar-4]);
					
				}
			}
			dirIdx++;
		}
		closedir(sourcedir);
		
		if (minTime != -1){
			snprintf(filepath, BUFSIZE, "%s%s", sourcedirpath, dirfilenames[minIdx]);
			printf("Earliest file is %s \n", filepath);
			snprintf(targetfilepath, BUFSIZE, "%s%s", targetdirpath, dirfilenames[minIdx]);
			printf("Moving file to %s \n", targetfilepath);
			MoveFile(filepath, targetfilepath);
		}
		else{
			printf("Nothing to move! \n");
		}
		
		// delete earliest in targetdir
		targetdir = opendir(targetdirpath);
		
		dirIdx = 0;
		minTime = -1;
		while((targetent = readdir(targetdir))!=NULL){
			strcpy(dirfilenames[dirIdx], targetent->d_name);
			countChar = snprintf(filepath, BUFSIZE, "%s%s", targetdirpath, dirfilenames[dirIdx]);
			filenamelen = countChar - targetdirpathlen;
			// printf("%s, filename length = %i \n", filepath, filenamelen);
		
			if(filenamelen > 10){
				fp = fopen(filepath, "rb");
				fseek(fp,0, SEEK_END);
				fileBytes = ftell(fp);
				fclose(fp);
				
				if (fileBytes == 200003584){
					sscanf(dirfilenames[dirIdx], "%d", &filenametimes[dirIdx]);
					// printf("Time of file = %i, size = %i bytes \n", filenametimes[dirIdx], fileBytes);

					
					if (minTime == -1){ // initialized
						minTime = filenametimes[dirIdx];
						minIdx = dirIdx;
					}
					else{
						if (filenametimes[dirIdx]<minTime){
							minTime = filenametimes[dirIdx];
							minIdx = dirIdx;
						}
					}
							
					// printf("Last four chars of file is %s \n", &filepath[countChar-4]);
					
				}
			}
			dirIdx++;
		}
		closedir(targetdir);
		printf("Total %i files in targetpath \n", dirIdx);
		
		if (dirIdx>storedBufferLen){
			snprintf(filepath, BUFSIZE, "%s%s", targetdirpath, dirfilenames[minIdx]);
			printf("Deleting earliest file is %s \n", filepath);
			DeleteFileA(filepath);
		}
		else{
			printf("Nothing to delete! \n");
		}
		
		Sleep(500);
	}

	return 0;
}