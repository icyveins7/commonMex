function [qf2_surface, cutoutNorm_dsSq, rxNorm_ds] = gen_xcorr(cutout, rx, selectedFreqRange, chnBW, searchFreq, downsampleRate, outputType, freqDS_factor)
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
    windowIdx = int32(find(freq>=selectedFreqRange(1) & freq<selectedFreqRange(2)));
    window_mask = zeros(1,length(rx_fft));
    window_mask(windowIdx) = 1;
    
    % Pre-window the cutout_fft
    cutout_fft_windowed = (cutout_fft .* window_mask).'; % col vector
%     cutout_fft_windowed_conj = conj(cutout_fft_windowed);
    
    % Pre-calculate downsampled indices (probably can find a better method)
    dsIdxMark = find(fftshift(1:length(rx_fft)) == length(rx_fft)/downsampleRate/2);
    dsIdx = int32( (dsIdxMark - length(rx_fft)/downsampleRate + 1) : dsIdxMark );
    fprintf(1,'Length of downsampled indices = %i\n', int32(length(dsIdx)));
    
    % Pre-downsample the cutout_fft
    cutout_fft_windowed_ds = cutout_fft_windowed(dsIdx);
    cutout_fft_windowed_conjds = conj(cutout_fft_windowed_ds);
    
    % Pre-calculate normalization
    cutoutNorm_dsSq = sum(cutout_fft_windowed_ds .* cutout_fft_windowed_conjds) / length(dsIdx); % remember to divide by N
    
    % Theoretical downsampled cutout length
    cutoutlen_ds = int32(length(cutout)/downsampleRate);
    fprintf(1, 'Length of downsampled cutout = %i\n', cutoutlen_ds);
    
    % Frequency downsampling factor
%     freqDS_factor = 4;
    fprintf(1, 'Skipping frequencies at %i bins/iteration, equivalent to %g Hz resolution.\n', freqDS_factor, chnBW/length(rx) * freqDS_factor);
    
    % Pre-calculate (still padded) downsampled cutout that will be used
    cutout_windowed_ds = ifft(ifftshift(cutout_fft_windowed_ds)) / (length(rx)/length(cutout_fft_windowed_ds)); % adjust IFFT factor for different N (FFT'ed at length(rx) but IFFT'ed at downsampled factor of that)
    keyboard;
    
    % Pre-calculate shape from cutout
    shape2match = prepareShapematch(cutout_windowed_ds(1:cutoutlen_ds), cutoutlen_ds, chnBW/downsampleRate, chnBW/downsampleRate); % we only prepare this using the exact length expected to generate the shape
    
    if ~searchFreq
        fprintf(1, 'Not implemented yet.\n');
        return;
        
        
    else % now we circular shift the frequencies
        numBins = length(windowIdx);
        numFreqIters = int32(floor( (length(rx_fft) - numBins + 1) / freqDS_factor));
        
%         if outputType == 0
%             qf2_surface = zeros(length(dsIdx)-cutoutlen_ds+1, numFreqIters);
%         elseif outputType == 1
%             % List of Outputs
%             % 1) Max Qf2
%             % 2) Max Qf2/Median Qf2
%             % 3) Shape Qf2
%             qf2_surface = zeros(3, numFreqIters, 'single');
%         end
%             
%         fprintf(1,'Entering freq iteration loop\n');
%         
%         for i = 1:numFreqIters
%             fprintf(1,'On freqIter %i\n', i);
%             atic = tic;
%             
%             rx_fft_windowed = zeros(length(rx_fft), 1); % column vector
%             rx_fft_windowed(windowIdx) = rx_fft((i-1)*freqDS_factor + 1:(i-1)*freqDS_factor+numBins);
%             rx_fft_windowed_ds = rx_fft_windowed(dsIdx);
%             
%             % Calculate preproduct
%             preproduct = cutout_fft_windowed_conjds .* rx_fft_windowed_ds;
% 
%             result = ifft(preproduct);
%             result = result(1:end-cutoutlen_ds+1); % clip to the useful parts
%             
%             % Calculate normalization
%             rx_windowed_ds = ifft(rx_fft_windowed_ds);
%             rxNorm_ds = zeros(length(result),1);
%             for k = 1:length(rxNorm_ds)
%                 rxNorm_ds(k) = norm(rx_windowed_ds(k:k+cutoutlen_ds-1));
%             end % end of rx_windowed_ds norm calc loops
%             
%             % Finally normalize and output the result as qf2
%             qf2 = abs(result).^2/cutoutNorm_dsSq./rxNorm_ds.^2;
%             
%             % And save it into the full matrix output
%             if outputType == 0
%                 qf2_surface(:,i) = qf2;
%             elseif outputType == 1
%                 % Convert to linear qf2
%                 linqf2 = qf2./(1-qf2);
%                 
%                 % Find maximum
%                 [maxq, maxqi] = max(qf2);
%                 
%                 % Find ratios
%                 medianq = median(linqf2);
%                 
%                 % Find shapeqf2
%                 shapeqf2 = getLinearShapeMatchQF2(shape2match(:), maxqi, qf2);
%                 
%                 % Arrange into the rows
%                 qf2_surface(1,i) = maxq;
%                 qf2_surface(2,i) = linqf2(maxqi)/medianq;
%                 qf2_surface(3,i) = shapeqf2;
%                 
%                 
%             end
% 
%             fprintf('Full loop time = %g s.\n', toc(atic));
%             
% %             % debugging plots
% %             keyboard;
% %             figure(101); clf; plot(qf2); hold on; plot(maxqi, maxq, 'rx'); plot(medianq + zeros(1,length(qf2)), 'r--'); title(['Shapeqf2 = ' num2str(shapeqf2)]);
% %             % end of debugging
%             
%         end % end of numFreqIters loop
        
        % mexfile ver
        tic;
        qf2_surface = mexIpp_gen_xcorr_inner(rx_fft, cutout_fft_windowed_conjds, dsIdx, windowIdx, freqDS_factor, numBins, cutoutlen_ds, numFreqIters, cutoutNorm_dsSq);
        toc;
        
    end % end of if/else for searchFreq condition

end