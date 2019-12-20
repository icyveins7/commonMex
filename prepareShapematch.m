function shape = prepareShapematch(orig, shapelen, chnBW, bw_signal)
    orig = orig(:); % ensure column vec

    x = [zeros(1,length(orig)) orig.' zeros(1,length(orig))]; % pad in front and back, note it is conjugated here
    
    % noise additions
    sigPwr = norm(orig)^2/length(orig); % expect that it may not be exact normalized, so adjust to exactly 1
    snr_inband = 1; % standardize to get 0.5 qf2 
    basicnoise = (randn(1,length(x)) + 1i*randn(1,length(x))) / sqrt(2) * sqrt(sigPwr) / sqrt(snr_inband) * sqrt(chnBW/bw_signal);
    xnoise = x + basicnoise;
    
    idx2slice = length(orig) + 1 - shapelen/2 : length(orig) + shapelen/2;

    [qn, ~] = dot_xcorr_singleChannel_2018_24(conj(orig), xnoise, cumsum(xnoise.*conj(xnoise)), norm(orig)^2, 1:length(orig)*2); % this should generate a peak ~0.5 qf2 shape

    shape = qn(idx2slice);
end