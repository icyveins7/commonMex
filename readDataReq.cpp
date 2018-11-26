#undef UNICODE

#define WIN32_LEAN_AND_MEAN

#include <process.h>
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include <zip.h>

#include <tchar.h>
#include <Shlwapi.h>

#pragma comment(lib, "User32.lib")
#pragma comment(lib, "Shlwapi.lib")

// Need to link with Ws2_32.lib
#pragma comment (lib, "Ws2_32.lib")
// #pragma comment (lib, "Mswsock.lib")

#define SAMP_RATE 5000 // change all these next time
#define MAX_FILES 128

#define DEFAULT_BUFLEN 512
#define DEFAULT_PORT "27015"

#define CREATED_ZIPFILE_NAME "testing.zip"

struct packet_contents{
	uint8_t dataType;
	uint32_t channelNumber;
	uint32_t startTime;
	uint32_t endTime;
};

int makeDataPaths(char *mainDirectoryPath, char *altDirectoryPath, uint32_t startTime, uint32_t endTime, char *allfilepaths, uint32_t channelNumber, uint8_t channelType){
	uint32_t totalTime = endTime - startTime + 1;
	const int suffixLen = 8;
	char channelStringSuffix[suffixLen * 4];
	
	// fill the suffix stringstream
	snprintf(&channelStringSuffix[2*suffixLen], suffixLen, "_sig"); // none for 0 or 1 so far..
	snprintf(&channelStringSuffix[3*suffixLen], suffixLen, "_msg");
	
	const int channelStringLen = 16;
	char channelString[channelStringLen];
	snprintf(channelString, channelStringLen, "%u_%u%s\\", channelNumber, channelType, &channelStringSuffix[channelType*suffixLen]); // have to make a special one for mainDirectoryPath

	for (int i = 0; i<totalTime; i++){ // FIX THE PATHING TO INCLUDE THE CHANNELNUMBER AND TYPE (_2 or _3) AS WELL THEN IT SHOULD WORK
		snprintf(&allfilepaths[i*3*MAX_PATH + 0 * MAX_PATH], MAX_PATH, "%s%s%u_channel.bin", mainDirectoryPath, channelString, startTime + (uint32_t)i);
		snprintf(&allfilepaths[i*3*MAX_PATH + 1 * MAX_PATH], MAX_PATH, "%s%s%u_allproductpeaks.bin", mainDirectoryPath, channelString, startTime + (uint32_t)i);
		snprintf(&allfilepaths[i*3*MAX_PATH + 2 * MAX_PATH], MAX_PATH, "%s%s%u_allfreqlistinds.bin", mainDirectoryPath, channelString, startTime + (uint32_t)i);
	
		printf("%s\n%s\n%s\n\n", &allfilepaths[i*3*MAX_PATH + 0 * MAX_PATH], &allfilepaths[i*3*MAX_PATH + 1 * MAX_PATH], &allfilepaths[i*3*MAX_PATH + 2 * MAX_PATH]);
	}
	
	return 0;
}

int zipFileList(char *filepaths, int len){
	long int firstNumber, lastNumber;
	int *errorp = NULL;
	zip_error_t *errorz = NULL;
	
	char filename[MAX_PATH];
	char endptr[MAX_PATH];
	
	zip_uint64_t zipfileIdx = 0;
	//zip_stat_t stat;
	
	strcpy(filename, filepaths); // get the first file time
	PathStripPathA(filename);
	firstNumber = strtol(filename, (char**)&endptr, 10);
	strcpy(filename, &filepaths[(len-1)*MAX_PATH]); // get the last file time
	PathStripPathA(filename);
	lastNumber = strtol(filename, (char**)&endptr, 10);
	
	char zipfilename[MAX_PATH];
	snprintf(zipfilename,MAX_PATH,CREATED_ZIPFILE_NAME);
	printf("Zipfilename is %s \n", zipfilename);
	
	zip_t *zipfile = zip_open(zipfilename, ZIP_CREATE, errorp);
	zip_source_t *zipsource;
	
	if (errorp == NULL){
		printf("Opened zip file for writing!\n");

		for (int i = 0; i<len; i++){
			strcpy(filename, &filepaths[i*MAX_PATH]);
			PathStripPathA(filename);
			printf("After stripping path from %s, filename is %s \n", &filepaths[i*MAX_PATH], filename);
			
			if((zipsource=zip_source_file_create(&filepaths[i*MAX_PATH], 0, -1, errorz)) == NULL || (zipfileIdx = zip_file_add(zipfile, filename, zipsource, ZIP_FL_OVERWRITE)) < 0){
				zip_source_free(zipsource);
				printf("Failed to add %s to zip! \n", filename);
				return 1;
			}
			else{
				zip_set_file_compression(zipfile, zipfileIdx, ZIP_CM_STORE, 1); // 1 to 9, 1 being fastest, ZIP_CM_DEFAULT, ZIP_CM_STORE
				printf("Set file compression to fastest!\n");
			}
		}
	}
	
	if (zip_close(zipfile) < 0){
		zip_discard(zipfile);
		printf("Failed to write zip file to disk! \n");
		return 1;
	}

	return 0;
}

int __cdecl main(void) 
{
    WSADATA wsaData;
    long int iResult;

    SOCKET ListenSocket = INVALID_SOCKET;
    SOCKET ClientSocket = INVALID_SOCKET;

    struct addrinfo *result = NULL;
    struct addrinfo hints;

    uint64_t iSendResult, totalSent;
	int checkDataPathResult;
	int checkZipResult;
	FILE *fp;
	uint64_t fileBytes;
	
    char recvbuf[DEFAULT_BUFLEN];
    int recvbuflen = DEFAULT_BUFLEN;
	
	// prep to make the filepaths
	char *longsendbuf = (char*)malloc(sizeof(int32_t)*MAX_FILES*SAMP_RATE*3); // there's going to be 2 for the complex data, 0.5 each for prodpeaks/freqlist_inds, since it's at half samp_rate
	char *altDirectoryPath = "C:\\DebugRTresults\\results\\";
	char *mainDirectoryPath = "C:\\DebugRTresults\\processedresults\\";
	char *allfilepaths = (char*)malloc(sizeof(char)*MAX_FILES*3*MAX_PATH); // 3 because there are 3 files to transfer (chan,prodpeaks,freqlist)
    
	// test the pathmaking function (seems like it's working)
	// makeDataPaths(mainDirectoryPath, altDirectoryPath, 1541011890, 1541011900, allfilepaths);
	
    // Initialize Winsock
    iResult = WSAStartup(MAKEWORD(2,2), &wsaData);
    if (iResult != 0) {
        printf("WSAStartup failed with error: %d\n", iResult);
        return 1;
    }

    ZeroMemory(&hints, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags = AI_PASSIVE;

    // Resolve the server address and port
    iResult = getaddrinfo(NULL, DEFAULT_PORT, &hints, &result);
    if ( iResult != 0 ) {
        printf("getaddrinfo failed with error: %d\n", iResult);
        WSACleanup();
        return 1;
    }

    // Create a SOCKET for connecting to server
    ListenSocket = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (ListenSocket == INVALID_SOCKET) {
        printf("socket failed with error: %ld\n", WSAGetLastError());
        freeaddrinfo(result);
        WSACleanup();
        return 1;
    }

    // Setup the TCP listening socket
    iResult = bind( ListenSocket, result->ai_addr, (int)result->ai_addrlen);
    if (iResult == SOCKET_ERROR) {
        printf("bind failed with error: %d\n", WSAGetLastError());
        freeaddrinfo(result);
        closesocket(ListenSocket);
        WSACleanup();
        return 1;
    }

    freeaddrinfo(result);

    iResult = listen(ListenSocket, SOMAXCONN);
    if (iResult == SOCKET_ERROR) {
        printf("listen failed with error: %d\n", WSAGetLastError());
        closesocket(ListenSocket);
        WSACleanup();
        return 1;
    }

    // Accept a client socket
	struct sockaddr_in their_addr;
	socklen_t addr_size = sizeof(their_addr);
    ClientSocket = accept(ListenSocket, (struct sockaddr *)&their_addr, &addr_size);
    if (ClientSocket == INVALID_SOCKET) {
        printf("accept failed with error: %d\n", WSAGetLastError());
        closesocket(ListenSocket);
        WSACleanup();
        return 1;
    }
	
	iResult = getpeername(ClientSocket, (struct sockaddr *)&their_addr, &addr_size);
	
	// char theirhostname[NI_MAXHOST];
	// char theirservInfo[NI_MAXSERV];
	// iResult = getnameinfo((struct sockaddr *)&their_addr, addr_size, theirhostname, NI_MAXHOST, theirservInfo, NI_MAXSERV, NI_NUMERICSERV);
	if (iResult != 0){
		printf("getpeername/getnameinfo failed with error # %ld \n", WSAGetLastError());
		WSACleanup();
		return 1;
	}
	// else{
		// printf("getpeername/getnameinfo returned hostname = %s  \n", theirhostname,);
	// }
	
	char *theirIPaddr;
	if ((theirIPaddr = inet_ntoa(their_addr.sin_addr))!=NULL){
		printf("%s is their IP address\n", theirIPaddr);
	}
	
	// if (inet_ntop(AF_INET, (void *)&their_addr, theirIPaddr, 64)!=NULL){
		// printf("%s is their IP address \n", theirIPaddr);
	// }

    // No longer need server socket
    closesocket(ListenSocket);

	// declare stuff to interpret packet
	struct packet_contents readPacket;
	
    // Receive until the peer shuts down the connection
    do {

        iResult = recv(ClientSocket, recvbuf, recvbuflen, 0);
        if (iResult > 0) {
            printf("Bytes received: %d\n", iResult);
			
			// attempt to interpret packet
			memcpy(&readPacket, recvbuf, sizeof(struct packet_contents));
			printf("Type number = %u \n", readPacket.dataType);
			printf("Channel number (hostbyteorder) = %u, (networkbyteorder) = %u \n", ntohl(readPacket.channelNumber), readPacket.channelNumber);
			printf("Start time (hostbyteorder) = %u, (networkbyteorder) = %u\n", ntohl(readPacket.startTime), readPacket.startTime); // probably just stick to hostbyteorder and don't convert on both sides
			printf("End Time (hostbyteorder) = %u, (networkbyteorder) = %u \n", ntohl(readPacket.endTime), readPacket.endTime);
			
			// attempt to make zip file based on packet
			makeDataPaths(mainDirectoryPath, altDirectoryPath, readPacket.startTime, readPacket.endTime, allfilepaths, readPacket.channelNumber, readPacket.dataType);
			checkZipResult = zipFileList(allfilepaths, ((int)readPacket.endTime-(int)readPacket.startTime+1)*3); 
			if (checkZipResult != 0){
				iResult = snprintf(longsendbuf, MAX_PATH, "FAILED_PACKET_CREATION"); // set the number of bytes to send to the string length
			}
			else{
				fp = fopen(CREATED_ZIPFILE_NAME,"rb");
				fseek(fp, 0, SEEK_END);
				fileBytes = ftell(fp); // get the size
				fseek(fp, 0, SEEK_SET); // go back to start
				memcpy(longsendbuf, &fileBytes, sizeof(uint64_t)); // put the file length in uint64 at the start to send
				fread(&longsendbuf[8],sizeof(char),fileBytes,fp); // read the whole file into the buffer after the 8 bytes used for the file length
				fclose(fp);
				iResult = fileBytes;
			}
			
			// Echo the send buffer back to the sender
			iSendResult = send( ClientSocket, longsendbuf, 8, 0); // send the length first in one packet
			if (iSendResult == SOCKET_ERROR) {
				printf("Initial send (packet length) failed with error: %d\n", WSAGetLastError());
				closesocket(ClientSocket);
				WSACleanup();
				return 1;
			}
			
			// now send the data itself
			totalSent = 8;
			while(totalSent < fileBytes + 8){
				iSendResult = send( ClientSocket, &longsendbuf[totalSent], fileBytes, 0 );
				if (iSendResult == SOCKET_ERROR) {
					printf("send failed with error: %d\n", WSAGetLastError());
					closesocket(ClientSocket);
					WSACleanup();
					return 1;
				}
				totalSent = totalSent + iSendResult;
				printf("Bytes sent: %llu\n", totalSent);
			}
        }
        else if (iResult == 0)
            printf("Connection closing...\n");
        else  {
            printf("recv failed with error: %d\n", WSAGetLastError());
            closesocket(ClientSocket);
            WSACleanup();
            return 1;
        }

    } while (iResult > 0);

    // shutdown the connection since we're done
    iResult = shutdown(ClientSocket, SD_SEND);
    if (iResult == SOCKET_ERROR) {
        printf("shutdown failed with error: %d\n", WSAGetLastError());
        closesocket(ClientSocket);
        WSACleanup();
        return 1;
    }

    // cleanup
    closesocket(ClientSocket);
    WSACleanup();
	free(allfilepaths);
	free(longsendbuf);

    return 0;
}