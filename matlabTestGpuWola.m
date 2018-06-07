fclose all; close all; clc;

% fid = fopen('50M_int16.bin','rb');
fid = fopen('400e3_int16.bin','rb');
data = fread(fid,inf,'int16=>int16');
% data = data(1:2:end)+1i.*data(2:2:end);
data = complex(data(1:2:end),data(2:2:end));
fclose(fid);

% fid = fopen('500k_ftap.bin','rb');
fid = fopen('5e3_ftap.bin','rb');
ftap = fread(fid,inf,'double');
fclose(fid);

% N = 10000;
% Dec = 10000;
N = 10;
Dec = 10;

gpuDevice();
out = gpuWola(data,ftap,N,Dec); % doesn't crash, looking good

% print some output
out(1,1:20).'

data = double(data);
% if you clear all, this line onwards will cause the mexfile ABOVE to crash
% repeatedly, saw this in SO forum as well
checkbasechannel = filter(ftap,1,data); % it crashes on this on the second f5, why?
checkbasechannel = checkbasechannel(1:Dec:end);

% visual check
[checkbasechannel(1:20) out(1,1:20).']

% max check
[m, ind] = max(checkbasechannel-out(1,:).');
checkbasechannel(ind)
out(1,ind)