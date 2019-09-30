// Tagged to ProcessMessagesRT Version 1.9.7
#include <iostream>
#include <string>
#include <windows.h>
#include <process.h>
#include <datetimeapi.h>
#include <tchar.h>
#include <Shlwapi.h>
#include <regex>

#pragma comment(lib, "User32.lib")
#pragma comment(lib, "Shlwapi.lib")

// use defines for the channel types (which are kinda fixed)
#define CHANNEL_TYPE_START 2
#define CHANNEL_TYPE_END 3
#define NUM_CHANNEL_TYPES 2

int main(int argc, char **argv){
	if (argc<6){
		std::cout<<"Arguments are (numChans) (delBackoff) (checkpointpath) (resultspath) (verboseFlag)"<<std::endl;
		return 1;
	}
	
    // read inputs
	int numChans;
	sscanf(argv[1], "%i", &numChans);
	
    int delBackoff;
	sscanf(argv[2], "%i", &delBackoff);
	
	std::string checkpointpath(argv[3]);
	
	std::string resultspath(argv[4]);

    int verboseFlag;
	sscanf(argv[5], "%i", &verboseFlag);
	
    // alloc arrays to compare
    int *checkpoints = (int*)calloc(numChans * NUM_CHANNEL_TYPES,sizeof(int));
    int *delpoints = (int*)calloc(numChans * NUM_CHANNEL_TYPES,sizeof(int));


    // loop counters
    int di; // deletion index
    int ci; // channel index

    // folder variables
    int folderExists;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	WIN32_FIND_DATA ffd;
	DWORD dwAttrs;
	std::string checkpointpath_dir = checkpointpath + "\\*";
	std::string channelcheckpointpath; // used for the specific channel
	std::string channelcheckpointpath_dir;
	std::string checkpoint; // used for the checkpoint file itself
	int channelType, channel;
	int checkpointValue;
	HANDLE hFindChan = INVALID_HANDLE_VALUE;
	WIN32_FIND_DATA ffdChan;
	DWORD dwAttrsChan;
	
	// checkpoint folder regex
	std::regex rx("\\d+_\\d+"); // we remove the newline char and use regex_search instead
	std::cmatch narrowMatch; // we use a C-style string.
	bool rxMatched;
	
	// deletion variables
	std::string delfilepath_chan;
	std::string delfilepath_pp;
	std::string delfilepath_flist;
	std::string channelresultspath;
	std::string channelresultspath_dir;
	std::string Delfile;
	std::string DelfileTime_str;
	HANDLE hFindDel = INVALID_HANDLE_VALUE;
	WIN32_FIND_DATA ffdDel;
	DWORD dwAttrsDel;
	int DelfileTime;
	
	// for err handling
	DWORD lastError;
	char lpMsgBuf[128];

	// print some initial checks
	std::cout<<"Searching "<<checkpointpath_dir<<std::endl;
	
    // main deletion loop
    while(true){
        // iterate over the checkpoint folders
		hFind = FindFirstFileA(checkpointpath_dir.c_str(), &ffd);
		FindNextFileA(hFind, &ffd); // read . and ..
		
		while(FindNextFileA(hFind,&ffd) != 0){
			// test folder with regex
			if (rxMatched = std::regex_search((const char*)ffd.cFileName, (const char*)(ffd.cFileName + strlen(ffd.cFileName)), narrowMatch, rx)){
				channelcheckpointpath = checkpointpath + "\\" + ffd.cFileName;
				channelcheckpointpath_dir = channelcheckpointpath + "\\*.chkpt"; // go straight to the checkpoint file

				if ((hFindChan = FindFirstFileA(channelcheckpointpath_dir.c_str(), &ffdChan))!=INVALID_HANDLE_VALUE){ 
					if (verboseFlag){std::cout<<"Found checkpoint file "<<ffdChan.cFileName<<" in "<<ffd.cFileName<<std::endl;}
				
					checkpoint = std::string(ffdChan.cFileName);
					checkpoint = checkpoint.substr(0,10); // clip to the first 10
					checkpointValue = std::stoi(checkpoint);
					
					// calculate the indices
					sscanf(ffd.cFileName, "%i_%i", &channel, &channelType);
					
					if (verboseFlag){
						std::cout<<"In "<<channelcheckpointpath<<" with checkpoint "<<checkpointValue<<std::endl;
						std::cout<<"Channel read is "<<channel<<" with type "<<channelType<<std::endl;
					}
					
					// shift the indices where necessary
					channelType = channelType - CHANNEL_TYPE_START;
					
					// obtain the array index
					ci = channel*NUM_CHANNEL_TYPES + channelType;
					
					// make the channelresultspaths
					channelresultspath = resultspath + "\\" + ffd.cFileName;
					channelresultspath_dir = channelresultspath + "\\*";
					
					// update the checkpoints array
					checkpoints[ci] = checkpointValue;
					
					// check if this is a new checkpoint i.e. new folder by using delpoints = 0 as condition
					if (delpoints[ci] == 0){
						std::cout<<"New checkpoint folder found in "<<channelcheckpointpath<<", going to delete from "<<channelresultspath<<std::endl;
						
						// if it is a new checkpoint then we iterate over the del folder
						hFindDel = FindFirstFileA(channelresultspath_dir.c_str(), &ffdDel);
						FindNextFileA(hFindDel, &ffdDel);
						
						while(FindNextFileA(hFindDel,&ffdDel) != 0){
							Delfile = std::string(ffdDel.cFileName);
							DelfileTime_str = Delfile.substr(0,10);
							DelfileTime = std::stoi(DelfileTime_str);
							Delfile = channelresultspath + "\\" + Delfile;
							
							if (DelfileTime < checkpoints[ci] - delBackoff){
								if (verboseFlag){
									std::cout<<"Deleting "<<Delfile<<std::endl;
								}
								if (DeleteFileA(Delfile.c_str()) == 0){
									lastError = GetLastError();
									FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, lastError, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), lpMsgBuf, 128 , NULL);
									std::cout<<"NEW CHECKPOINT SCAN: Failed to delete "<<Delfile<<", CODE: "<<lastError<<", "<<lpMsgBuf<<std::endl;	
								}
							}
							
						}
						FindClose(hFindDel);
						
					}
					else{ // if it's not new then just use the old delpoint -> new checkpoint
						for (di = delpoints[ci]; di < checkpoints[ci] - delBackoff; di++){
							delfilepath_chan = channelresultspath + "\\" + std::to_string(di) + "_channel.bin";
							delfilepath_pp = channelresultspath + "\\" + std::to_string(di) + "_allproductpeaks.bin";
							delfilepath_flist = channelresultspath + "\\" + std::to_string(di) + "_allfreqlistinds.bin";
							
							if (verboseFlag){
								std::cout<<delfilepath_chan<<std::endl<<delfilepath_pp<<std::endl<<delfilepath_flist<<std::endl;
							}
							
							// perform the deletions
							if (DeleteFileA(delfilepath_chan.c_str()) == 0){
								lastError = GetLastError();
								FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, lastError, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), lpMsgBuf, 128 , NULL);
								if(verboseFlag){std::cout<<"Failed to delete "<<delfilepath_chan<<", CODE: "<<lastError<<", "<<lpMsgBuf<<std::endl;		}
							}
							if (DeleteFileA(delfilepath_pp.c_str()) == 0){
								lastError = GetLastError();
								FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, lastError, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), lpMsgBuf, 128 , NULL);
								if(verboseFlag){std::cout<<"Failed to delete "<<delfilepath_pp<<", CODE: "<<lastError<<", "<<lpMsgBuf<<std::endl;	}
							}
							if (DeleteFileA(delfilepath_flist.c_str()) == 0){
								lastError = GetLastError();
								FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, lastError, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), lpMsgBuf, 128 , NULL);
								if(verboseFlag){std::cout<<"Failed to delete "<<delfilepath_flist<<", CODE: "<<lastError<<", "<<lpMsgBuf<<std::endl;	}
							}
						}
					}
					
					// regardless of what happens, we just update the new delpoint
					delpoints[ci] = checkpoints[ci] - delBackoff;
					
					FindClose(hFindChan);
				}
				else{
					std::cout<<"Could not find checkpoint in folder "<<ffd.cFileName<<std::endl;
				}
				
				// at the end of each checkpoint folder print the updated values in the arrays?
				std::cout<<"=== CHANNEL: "<<channel<< ", CHANNEL_TYPE: "<<channelType + NUM_CHANNEL_TYPES<<std::endl;
				std::cout<<"=== CI: "<<ci<<", CHKPT_ARR[CI]: "<<checkpoints[ci]<<", DELPT_ARR[CI]: "<<delpoints[ci]<<std::endl;

			}
			else{
				std::cout<<ffd.cFileName<<" is not a valid folder"<<std::endl;
			}
		}
		FindClose(hFind);
		
		// wait a little?
		Sleep(1000);
    } // end of while loop
    
    // cleanup
    free(checkpoints);
    free(delpoints);

    return 0;
}
