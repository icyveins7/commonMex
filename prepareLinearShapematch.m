function shape = prepareLinearShapematch(orig, shapelen)
    orig = orig(:); % ensure column vec

    x = [zeros(1,length(orig)) orig.' zeros(1,length(orig))]; % pad in front and back, note it is conjugated here
    
    % noise additions
    sigPwr = norm(orig)^2/length(orig);
    snr_inband = 100;
    chnBW = 8e3;
    bw_signal = 8e3;
    basicnoise = (randn(1,length(x)) + 1i*randn(1,length(x))) / sqrt(2) * sqrt(sigPwr) / sqrt(snr_inband) * sqrt(chnBW/bw_signal);
    xnoise = x + basicnoise;
    
    idx2slice = length(orig) + 1 - shapelen/2 : length(orig) + shapelen/2;

    [q, ~] = dot_xcorr_singleChannel_2018_24(conj(orig), x, cumsum(x.*conj(x)), norm(orig)^2, 1:length(orig)*2);
    [qn, ~] = dot_xcorr_singleChannel_2018_24(conj(orig), xnoise, cumsum(xnoise.*conj(xnoise)), norm(orig)^2, 1:length(orig)*2);
    
    lq = q./(1-q); % make the linear qf2
    
    shape = lq(idx2slice);
    keyboard;
end