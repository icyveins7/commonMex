% this is different from eff SNR. used when xcorring a pure signal with one
% with noise, obtains the SNR for the noisy signal.
function SNR = convertQF2toSNR(qf2)
    SNR = qf2./(1.0-qf2);
end