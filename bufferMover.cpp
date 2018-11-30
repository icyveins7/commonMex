#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <windows.h>
#include <process.h>
#include <datetimeapi.h>
#include <tchar.h>
#include <Shlwapi.h>

#pragma comment(lib, "User32.lib")
#pragma comment(lib, "Shlwapi.lib")

const int64_t UNIX_TIME_START = 0x019DB1DED53E8000;
const int64_t TICKS_PER_SECOND = 10000000; // yes, 7 zeroes

void checkAlreadyMoved(int64_t optimalStartUnixSec, int64_t optimalEndUnixSec, uint8_t *toMove_boolVec, char *destmaindirectorypath, uint8_t *channelcopyswitch){
	char destfilepath[MAX_PATH]; // buffer for the full path to each binfile
	char destsubdirectory[MAX_PATH]; // buffer for the directory path for reading
	
	int i; // index for recording channel (should be 0 to 3)
	int64_t t, filetime; // index for time of the file, and filetime
	
	for (t=0; t<optimalEndUnixSec-optimalStartUnixSec; t++){
		filetime = t + optimalStartUnixSec;
		for (i=0; i<4; i++){
			if (channelcopyswitch[i] == 1){
				snprintf(destsubdirectory,MAX_PATH,"%s%i\\",destmaindirectorypath,i); // make the subdirectory path to 0,1,2,3
				snprintf(destfilepath,MAX_PATH,"%s%lli.bin",destsubdirectory,filetime); // make the full path to the file
				// printf("%s \n", destfilepath); // for debugging
				
				// if the file exists at destination then we don't flag for moving
				if (PathFileExistsA(destfilepath)){
					toMove_boolVec[t] = 0;
					printf("%s already present in storage \n", destfilepath); // for debugging
				}
				else{ 
					toMove_boolVec[t] = 1;
					// printf("%s to be moved! \n", destfilepath); // for debugging
				}
			}
		}
	}
}

void moveWhatIsNeeded(int64_t optimalStartUnixSec, int64_t optimalEndUnixSec, uint8_t *toMove_boolVec, char *destmaindirectorypath, uint8_t *channelcopyswitch, char *sourcemaindirectorypath){
	char destfilepath[MAX_PATH]; // buffer for the full path to each binfile
	char destsubdirectory[MAX_PATH]; // buffer for the directory path for writing
	char sourcefilepath[MAX_PATH]; // buffer for full path to source bin file
	char sourcesubdirectory[MAX_PATH]; // buffer for source directory path for reading
	
	DWORD lastError;
	char lpMsgBuf[128];
	
	int i; // index for recording channel (should be 0 to 3)
	int64_t t, filetime; // index for time of the file, and filetime
	
	// before everything, make the directories if they don't yet exist (likely for the first time)
	if (PathFileExistsA(destmaindirectorypath) == false){
		printf("Making destination main directory for first time \n");
		CreateDirectoryA(destmaindirectorypath, NULL);
	}
	for (i=0; i<4; i++){ // and these are for the subdirectories
		if (channelcopyswitch[i] == 1){
			snprintf(destsubdirectory,MAX_PATH,"%s%i\\",destmaindirectorypath,i); // make the subdirectory path to 0,1,2,3
			if (PathFileExistsA(destsubdirectory) == false){
				printf("Making destination subdirectory %i for first time \n", i);
				CreateDirectoryA(destsubdirectory, NULL);
			}
		}
	}
	
	// now we iterate over the times
	for (t=0; t<optimalEndUnixSec-optimalStartUnixSec; t++){
		filetime = t + optimalStartUnixSec;
		for (i=0; i<4; i++){
			if (channelcopyswitch[i] == 1){
				snprintf(destsubdirectory,MAX_PATH,"%s%i\\",destmaindirectorypath,i); // make the subdirectory path to 0,1,2,3
				snprintf(destfilepath,MAX_PATH,"%s%lli.bin",destsubdirectory,filetime); // make the full path to the file
				// printf("%s \n", destfilepath); // for debugging
				
				snprintf(sourcesubdirectory,MAX_PATH,"%s%i\\",sourcemaindirectorypath,i);
				snprintf(sourcefilepath,MAX_PATH,"%s%lli.bin",sourcesubdirectory,filetime);
				
				if (toMove_boolVec[t] == 1){ // then we have to move it
					if (MoveFile(sourcefilepath,destfilepath) == 0){
						lastError = GetLastError();
						printf("Failed to move %s with error code %i \n", sourcefilepath, lastError);
						FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, lastError, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), lpMsgBuf, 128 , NULL);
						printf("Error string : %s \n", lpMsgBuf);
					}
					else{
						printf("Successfully moved %s to %s \n",sourcefilepath,destfilepath);
					}
				}
			}
		}
	}	
	
}

int parseDatetimeFile(char *sourcemaindirectorypath, char *destmaindirectorypath, uint8_t *channelcopyswitch, int frontBuffer, int len2move){
	FILE *fp;
	fp = fopen("datetimes_to_store.txt","r");
	
	// declare everything needed
	// int len2move = 10; // 3 hours of data to move
	struct _SYSTEMTIME systemtimestruct;
	struct _FILETIME filetimestruct;
	char datestring[128];
	char timestring[128];
	LARGE_INTEGER li;
	int64_t unixseconds;
	int64_t optimalStartUnixSec, optimalEndUnixSec;
	int dateret, timeret, ret;
	uint8_t inputConfirm;
	uint8_t *toMove_boolVec = (uint8_t*)malloc(sizeof(uint8_t)*len2move); // 3 hours of data
	
	printf("\nPlease check that the datetimes are as desired\n\n");
	
	// loop over the whole file
	char line[256];
	fgets(line,sizeof(line),fp); // read the first line (the help line)
	while(fgets(line,sizeof(line),fp)){		
		// process each line
		sscanf(line, "%hu/%hu/%hu,%hu:%hu:%hu",&systemtimestruct.wYear,&systemtimestruct.wMonth,&systemtimestruct.wDay,&systemtimestruct.wHour,&systemtimestruct.wMinute,&systemtimestruct.wSecond);
		
		dateret = GetDateFormatEx(LOCALE_NAME_INVARIANT, DATE_LONGDATE, &systemtimestruct, NULL, (LPWSTR)datestring, sizeof(datestring), NULL);
		// if (dateret == 0){printf("Failed to get date format!\n");}
		timeret = GetTimeFormatEx(LOCALE_NAME_INVARIANT, TIME_FORCE24HOURFORMAT, &systemtimestruct, NULL, (LPWSTR)timestring, sizeof(timestring));
		// if (timeret == 0){printf("Failed to get time format!\n");}
		printf("Datetime is %ls, %ls \n",(LPWSTR)datestring, (LPWSTR)timestring); // print for confirmation
		
		ret = SystemTimeToFileTime(&systemtimestruct, &filetimestruct);
		printf("Enter 0 to confirm, any other number to stop program: ");
		scanf("%hhu",&inputConfirm);
		
		
		if (ret!=0 && inputConfirm == 0){
			li.LowPart = filetimestruct.dwLowDateTime;
			li.HighPart = filetimestruct.dwHighDateTime;
			
			unixseconds = (li.QuadPart - UNIX_TIME_START) / TICKS_PER_SECOND;
			unixseconds = unixseconds - 28800; // move 8 hours down to unix time
			
			printf("Converted time is %lli \n", unixseconds);
			
			optimalEndUnixSec = unixseconds + (int64_t)frontBuffer; // let's add 5 minutes for safety (clock differences between different systems?)
			optimalStartUnixSec = optimalEndUnixSec - len2move; // move 3 hours back in total
			
			memset(toMove_boolVec, 0, sizeof(uint8_t)*len2move); // zero it out
			checkAlreadyMoved(optimalStartUnixSec, optimalEndUnixSec, toMove_boolVec, destmaindirectorypath, channelcopyswitch); // check whether it's already in storage, flag the missing ones in the uint8 vector
			
			moveWhatIsNeeded(optimalStartUnixSec, optimalEndUnixSec, toMove_boolVec, destmaindirectorypath, channelcopyswitch, sourcemaindirectorypath); // now actually do the moving
		}
		else{
			printf("Failed to convert time\n");
		}
	}
	
	
	
	fclose(fp);
	return 0;
}

int main(int argc, char* argv[]){
	
	if (argc != 6){
		printf("Args are (sourcemaindirpath) (destmaindirpath) (channelcopystring) (frontBuffer) (totalSecsToSave)\n");
		return 1;
	}
	
	char sourcemaindirectorypath[MAX_PATH];
	char destmaindirectorypath[MAX_PATH];
	char channelcopystring[MAX_PATH];
	uint8_t channelcopyswitch[4];
	int frontBuffer, totalSecsToSave;
	
	snprintf(sourcemaindirectorypath,MAX_PATH,"%s",argv[1]);
	snprintf(destmaindirectorypath,MAX_PATH,"%s",argv[2]);
	snprintf(channelcopystring,MAX_PATH,"%s",argv[3]);
	sscanf(argv[4],"%i",&frontBuffer);
	sscanf(argv[5],"%i",&totalSecsToSave);
	for (int i=0;i<4;i++){
		if (channelcopystring[i] == '1'){
			channelcopyswitch[i] = 1;
		}
		else{ channelcopyswitch[i] = 0;}
	}
	
	parseDatetimeFile(sourcemaindirectorypath,destmaindirectorypath,channelcopyswitch, frontBuffer, totalSecsToSave);
	
	
	
	
	return 0;
}