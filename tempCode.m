% channelIdx = 13;
% xcorrIter = 10;

% startIdx = (xcorrIter-1) * cutoutlen + initStartIdx;
% startTime = (startIdx - 1) / chnBW;
% shifts = startIdx + idxshifts(1):1:startIdx + idxshifts(2); 
% tdvec = (shifts-1)/chnBW - startTime;
% cutout = channels{idx0}(channelIdx,startIdx:startIdx+cutoutlen-1); % taken from idx0

% conjcutout = conj(cutout);
% cutout_pwr = sum(cutout.*conj(cutout));
% othercutout = channels{idx1}(channelIdx,startIdx:startIdx+cutoutlen-1);
% norm1 = norm(othercutout);

% decFactor = 1000;
% pdt = conjcutout.*othercutout;
% pdtfft = fft(pdt);
% pdtfftzeroed = [pdtfft(1:10) zeros(1,length(pdtfft)-20) pdtfft(end-9:end)];
% pdtfftzeroed_dec0 = ifft(pdtfftzeroed); pdtfftzeroed_dec0 = pdtfftzeroed_dec0(1:decFactor:end); % from fs = 2e6 to 2e3
% % pdtfftzeroed_dec0_fft = fft(pdtfftzeroed_dec0);
% pdtfftzeroed_dec0_pad = [pdtfftzeroed_dec0 zeros(1,990)]; % pad to 1000
% pdtfftzeroed_dec0_pad_fft = fft(pdtfftzeroed_dec0_pad);
% figure; plot(abs(pdtfftzeroed_dec0_pad_fft).^2/cutout_pwr/norm1^2 * decFactor^2)

% % original pad
% pdtpad = [pdt zeros(1,1e6-length(pdt))];
% pdtpadfft = fft(pdtpad);
% figure; plot(abs(pdtpadfft).^2/cutout_pwr/norm1^2)

% % original nonpad
% figure; plot(abs(pdtfft).^2/cutout_pwr/norm1^2);


%% CZT OPTIMIZATION
numTests = 1000;

ylong = [yy zeros(1,padlen - length(yy))];
tic;
for i = 1:numTests
	t1 = fft(ylong);
end
toc;

fs = 2e6; f1 = -2e3; f2 = 2e3+5;       % In hertz
n = 801;
wo = exp(-j*2*pi*(f2-f1)/(n*fs));
w = -1i*2*pi*(f2-f1)/(n*fs);
ao = exp(j*2*pi*f1/fs);
a = 1i*2*pi*f1/fs;
nfft = 21000;

tic;
for i = 1:numTests
	t2 = czt(yy,n,wo,ao);
end
toc;

tic;
[premul_ww, premul_fv, premul_aa] = czt_opt_prep(yy, n, w, a, nfft);
for i = 1:numTests
	t3 = czt_opt(yy,n,premul_ww,premul_aa,nfft,premul_fv);
end
toc;