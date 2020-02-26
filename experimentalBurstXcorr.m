clear all; close all; clc;

fs = 1e5;

period = 1e4;
burstlen = 9000;
guardlen = 1000;

power = 10.0;
sig = randn(1,fs) + 1i*randn(1,fs);
sig = sig/sqrt(2) * sqrt(power);
numPeriods = length(sig)/period;
burstStartStops = zeros(2, numPeriods);
for i = 1:numPeriods
    burstStartStops(1,i) = (i-1)*period+1;
    burstStartStops(2,i) = (i-1)*period+burstlen;
end

mask = mexMakeMasksForUnknowns(length(sig),burstlen,guardlen,int32(0));
psig = sig.*mask.';

figure; plot(abs(psig));

t_off = 1000;
rx = [zeros(1,t_off) psig zeros(1,t_off)];
tone = mexIppTone_2018(length(rx), 1e3, fs);
shifted = rx.*tone.';

%% orig fft check
cutout = shifted(t_off+1:t_off+length(psig));

pdt = conj(cutout).*psig;
fftpdt = fft(pdt);

figure; plot(makeFreq(length(fftpdt),fs),abs(fftpdt).^2/norm(conj(cutout))^2 / norm(psig)^2);

%% now write a xcorr by slicing each burst

for i = 1:numPeriods
    
    
    
end