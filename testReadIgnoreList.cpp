#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void readIgnoredChannelListFile(char* path2file, int *ignoredChannels) {
	FILE *fp;
	fp = fopen(path2file, "r");
	char line[32];
	char cutline1[16];
	char cutline2[16];

	int startidx, endidx, channelNumber;

	int i;
	int colonExists;

	printf("Ignoring the following channels : \n");

	while (fgets(line, sizeof(line), fp)) {
		i = 0;
		colonExists = 0;
		while (i < 32) {
			if (line[i] == ':') { colonExists = 1; break; }
			i = i + 1;
		}

		if (colonExists) {
			memcpy(cutline1, &line[0], sizeof(char)*i);
			cutline1[i] = '\0';
			sscanf(cutline1, "%d", &startidx);

			memcpy(cutline2, &line[i + 1], sizeof(char)*(32 - i - 1));
			cutline2[strcspn(cutline2, "\n")] = '\0'; // change the newline to the terminating null
			sscanf(cutline2, "%d", &endidx);

			for (int k = startidx; k < endidx; k++) {
				ignoredChannels[k] = 1;
				printf("%i\n", k);
			}
		}
		else {
			sscanf(line, "%d", &channelNumber); // convert to int

			ignoredChannels[channelNumber] = 1;
			printf("%i\n", channelNumber);
		}
	}
	fclose(fp);
}

int main(){
	int *ignoredChannels = (int*)malloc(sizeof(int)*20000);
	
	char path2file[256];
	scanf("%s", path2file);
	readIgnoredChannelListFile(path2file, ignoredChannels);
	
	free(ignoredChannels);
	
	return 0;
}