#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#define SAMP_RATE 5000 // change all these next time
#define MAX_FILES 128

// Need to link with Ws2_32.lib, Mswsock.lib, and Advapi32.lib
#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")

#define DEFAULT_BUFLEN 512
#define DEFAULT_PORT "27015"

struct packet_contents{
	uint8_t dataType;
	uint32_t channelNumber;
	uint32_t startTime;
	uint32_t endTime;
};

int __cdecl main(int argc, char **argv) 
{
    WSADATA wsaData;
    SOCKET ConnectSocket = INVALID_SOCKET;
    struct addrinfo *result = NULL,
                    *ptr = NULL,
                    hints;
    // char *sendbuf = "this is a test";
	struct packet_contents packet0;
	packet0.dataType = 2;
	packet0.channelNumber = htonl(536);
	packet0.startTime = htonl(1541036538);
	// printf("host byte order = %u, network byte order = %u \n", 1541036538, htonl(1541036538));
	packet0.endTime = htonl(1541036558);
	char sendbuf1[DEFAULT_BUFLEN];
	memcpy(sendbuf1, &packet0, sizeof(struct packet_contents));
	
	struct packet_contents packet1;
	packet1.dataType = 3;
	packet1.channelNumber = 532;
	packet1.startTime = 1541011890;
	packet1.endTime = 1541011900;
	char sendbuf2[DEFAULT_BUFLEN];
	memcpy(sendbuf2, &packet1, sizeof(struct packet_contents));
	
    char *recvbuf = (char*)malloc(sizeof(int32_t)*MAX_FILES*SAMP_RATE*3); // same as readDataReq.cpp
    uint64_t iResult, totalReceived, expectedPacketSize;
	
	// make the workspace directory
	char workspacepath[MAX_PATH];
	GetCurrentDirectory(MAX_PATH, workspacepath);
	snprintf(workspacepath, MAX_PATH, "%s\\workspace\\", workspacepath);
	bool dirResult = CreateDirectoryA(workspacepath,NULL);
	
	// make the buffer for consequent directories
	char dirpath[MAX_PATH];
	
    // int recvbuflen = DEFAULT_BUFLEN;
	
	FILE *fp;
    
    // Validate the parameters
    if (argc != 2) {
        printf("usage: %s server-name\n", argv[0]);
        return 1;
    }

    // Initialize Winsock
    iResult = WSAStartup(MAKEWORD(2,2), &wsaData);
    if (iResult != 0) {
        printf("WSAStartup failed with error: %llu\n", iResult);
        return 1;
    }

    ZeroMemory( &hints, sizeof(hints) );
    hints.ai_family = AF_INET; // AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    // Resolve the server address and port
    iResult = getaddrinfo(argv[1], DEFAULT_PORT, &hints, &result);
    if ( iResult != 0 ) {
        printf("getaddrinfo failed with error: %llu\n", iResult);
        WSACleanup();
        return 1;
    }

    // Attempt to connect to an address until one succeeds
    for(ptr=result; ptr != NULL ;ptr=ptr->ai_next) {
		printf("Iterating through linked list of available address/port combinations..\n");
        // Create a SOCKET for connecting to server
        ConnectSocket = socket(ptr->ai_family, ptr->ai_socktype, 
            ptr->ai_protocol);
        if (ConnectSocket == INVALID_SOCKET) {
            printf("socket failed with error: %ld\n", WSAGetLastError());
            WSACleanup();
            return 1;
        }

        // Connect to server.
        iResult = connect( ConnectSocket, ptr->ai_addr, (int)ptr->ai_addrlen);
        if (iResult == SOCKET_ERROR) {
            closesocket(ConnectSocket);
            ConnectSocket = INVALID_SOCKET;
            continue;
        }
        break;
    }

    freeaddrinfo(result);

    if (ConnectSocket == INVALID_SOCKET) {
        printf("Unable to connect to server after finishing the linked list!\n");
        WSACleanup();
        return 1;
    }

	for (int ii = 1; ii < 2; ii++){
		// Send an initial buffer
		// iResult = send( ConnectSocket, sendbuf, (int)strlen(sendbuf), 0 );
		if (ii%2==0){
			iResult = send( ConnectSocket, sendbuf1, sizeof(struct packet_contents), 0); // let's test with 5 bytes first
		}
		else{
			iResult = send( ConnectSocket, sendbuf2, sizeof(struct packet_contents), 0);
		}
		if (iResult == SOCKET_ERROR) {
			printf("send failed with error: %d\n", WSAGetLastError());
			closesocket(ConnectSocket);
			WSACleanup();
			return 1;
		}

		printf("Bytes Sent in packet %i: %llu\n", ii, iResult);
		Sleep(1000);
	}

    // shutdown the connection since no more data will be sent
    iResult = shutdown(ConnectSocket, SD_SEND);
    if (iResult == SOCKET_ERROR) {
        printf("shutdown failed with error: %d\n", WSAGetLastError());
        closesocket(ConnectSocket);
        WSACleanup();
        return 1;
    }

    // Receive until the peer closes the connection
	totalReceived = 0;
    do {

        iResult = recv(ConnectSocket, &recvbuf[totalReceived], 65536, 0); // max IP packet length, though it will never hit that
		if ( totalReceived == 0){ // then it's the first packet telling you the length..
			memcpy(&expectedPacketSize,recvbuf, sizeof(uint64_t));
			printf("INCOMING PACKET WITH %llu BYTES \n", expectedPacketSize);
		}
        if ( iResult > 0 ){
            // printf("Bytes received: %d\n", iResult);
			totalReceived = totalReceived + iResult;
		}
        else if ( iResult == 0 ){ printf("Connection closed\n"); }
        else{printf("recv failed with error: %d\n", WSAGetLastError()); }

    } while( iResult > 0 );
	printf("TOTAL bytes received: %llu \n", totalReceived);
	if (totalReceived - 8 == expectedPacketSize){
		printf("FULL PACKET RECEIVED. WRITING TO FILE\n");
		fp = fopen("testing.zip","wb");
		fwrite(&recvbuf[8], sizeof(char), expectedPacketSize, fp);
		fclose(fp);
		
		// create the channel workspace directory
		snprintf(dirpath, MAX_PATH, "%s%u_%u\\", workspacepath, packet1.channelNumber, packet1.dataType);
		CreateDirectoryA(dirpath, NULL);
		printf("unzipping to %s \n", dirpath);
		
		// unzip the file to the directory
		
	}
	else{
		printf("FAILED TO RECEIVE FULL PACKET \n");
	}

    // cleanup
    closesocket(ConnectSocket);
    WSACleanup();

    return 0;
}