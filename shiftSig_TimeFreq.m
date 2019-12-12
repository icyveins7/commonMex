function shifted = shiftSig_TimeFreq(sig, t_shift, f_shift, sampleRate)

    % first shift the freq by
    sigfreqshift = sig.*exp(1i*2*pi*f_shift*(0:length(sig)-1)/sampleRate);
    
    % then go to fourier to shift time
    fftsig = fft(sigfreqshift);
    freqVec = makeFreq(length(sig), sampleRate);
    fftsigtimeshift = fftsig.*exp(1i*2*pi.*freqVec*-t_shift);
    
    % the final result
    shifted = ifft(fftsigtimeshift);
end