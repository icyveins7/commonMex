function [qf2_surface, cutoutNorm_ds, rxNorm_ds] = gen_xcorr(cutout, rx, selectedFreqRange, chnBW, searchFreq, downsampleRate)
    % Preprocess to make cutout same length as rx
    cutout_ext = [cutout zeros(1,length(rx)-length(cutout))];
    
    % Preprocess to get the ffts required
    cutout_fft = fft(cutout_ext);
    rx_fft = fft(rx);
    
    % Create extended freq vector
    freq = makeFreq(length(cutout_ext), chnBW);
    
    % Work in the centred frequency i.e. fftshifted, so that we can
    % circular shift later on
    freq = fftshift(freq);
    cutout_fft = fftshift(cutout_fft);
    rx_fft = fftshift(rx_fft);
    
    % Find the indices for the window of frequencies
    windowIdx = find(freq>=selectedFreqRange(1) & freq<selectedFreqRange(2));
    window_mask = zeros(1,length(rx_fft));
    window_mask(windowIdx) = 1;
    
    % Pre-window the cutout_fft
    cutout_fft_windowed = (cutout_fft .* window_mask).'; % col vector
    cutout_fft_windowed_conj = conj(cutout_fft_windowed);
    
    % Pre-calculate downsampled indices (probably can find a better method)
    dsIdxMark = find(fftshift(1:length(rx_fft)) == length(rx_fft)/downsampleRate/2);
    dsIdx = (dsIdxMark - length(rx_fft)/downsampleRate + 1) : dsIdxMark;
    fprintf(1,'Length of downsampled indices = %i\n', int32(length(dsIdx)));
    
    % Pre-downsample the cutout_fft
    cutout_fft_windowed_conjds = cutout_fft_windowed_conj(dsIdx);
    
    % Pre-calculate normalization
    cutoutNorm_ds = norm(cutout_fft_windowed_conjds);
    
    % Theoretical downsampled cutout length
    cutoutlen_ds = int32(length(cutout)/downsampleRate);
    fprintf(1, 'Length of downsampled cutout = %i\n', cutoutlen_ds);
    
    if ~searchFreq
        fprintf(1, 'Not implemented yet.\n');
        return;
        
        
    else % now we circular shift the frequencies
        numBins = length(windowIdx);
        numFreqIters = length(dsIdx) - numBins + 1;
        
        qf2_surface = zeros(length(dsIdx)-cutoutlen_ds+1, numFreqIters);
        
        fprintf(1,'Entering freq iteration loop\n');
        
        parfor i = 1:numFreqIters
            rx_fft_windowed = zeros(length(rx_fft), 1); % column vector
            rx_fft_windowed(windowIdx) = rx_fft(i:i+numBins-1);
            rx_fft_windowed_ds = rx_fft_windowed(dsIdx);
            
            % Calculate preproduct
            preproduct = cutout_fft_windowed_conjds .* rx_fft_windowed_ds;
            
            result = ifft(preproduct);
            result = result(1:end-cutoutlen_ds+1); % clip to the useful parts
            
            % Calculate normalization
            rx_windowed_ds = ifft(rx_fft_windowed_ds);
            rxNorm_ds = zeros(length(result),1);
            for k = 1:length(rxNorm_ds)
                rxNorm_ds(k) = norm(rx_windowed_ds(k:k+cutoutlen_ds-1));
            end % end of rx_windowed_ds norm calc loops
            
            % Finally normalize and output the result as qf2
            qf2 = abs(result./cutoutNorm_ds./rxNorm_ds).^2;
            
            % And save it into the full matrix output
            qf2_surface(:,i) = qf2;
            
        end % end of numFreqIters loop
        
        
        
    end % end of if/else for searchFreq condition

end