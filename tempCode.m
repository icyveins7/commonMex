pdtfft = fft(pdt);
pdtfftzeroed = [pdtfft(1:5) zeros(1,length(pdtfft)-10) pdtfft(end-4:end)];
pdtfftzeroed_dec0 = ifft(pdtfftzeroed); pdtfftzeroed_dec0 = pdtfftzeroed_dec0(1:2000:end); % from fs = 2e6 to 1e3
% pdtfftzeroed_dec0_fft = fft(pdtfftzeroed_dec0);
pdtfftzeroed_dec0_pad = [pdtfftzeroed_dec0 zeros(1,990)]; % pad to 1e3
pdtfftzeroed_dec0_pad_fft = fft(pdtfftzeroed_dec0_pad);
figure; plot(abs(pdtfftzeroed_dec0_pad_fft).^2/cutout_pwr/norm1^2 * 2000^2)