% To be used after the sample-level TD/FD correlation done with dot_xcorr or test_complex.
function [finefreqfound,timediff] = finerFreqTimeSearch(x_aligned, y_aligned, fineRes, freqfound, freqRes, fs, td_scan_range, steeringvec, debugPlotsMode)
	freq = makeFreq(length(x_aligned),fs);
	% === Finer Frequency Resolution (Multi Step) ===
	for i = 1:length(fineRes)
		fineFreq = freqfound-freqRes:fineRes(i):freqfound+freqRes;
		freqRes = fineRes(i);
        % matlab call
% 		fineshifts = exp(1i*2*pi*(-fineFreq.')*(0:length(x_aligned)-1)/fs); % note that it's -Freq, not +Freq, in the tone used to shift!
        % use mex call?
        fineshifts = mexIppTone_2018(length(x_aligned),-fineFreq,fs);
%         fineshifts = zeros(length(fineFreq),length(x_aligned));
%         for ff = 1:length(fineFreq)
%             fineshifts(ff,:) = mexIppTone_2018(length(x_aligned),-fineFreq(ff),fs);
%         end
%         keyboard;
% 		pp = zeros(1,length(fineFreq));
        precomputed = conj(y_aligned).*x_aligned;
        pp = precomputed*fineshifts;
% 		for l = 1:length(fineFreq)
% % 			pp(l) = sum(conj(y_aligned).*x_aligned.*fineshifts(l,:));
% %             pp(l) = sum(conj(y_aligned).*x_aligned.*fineshifts(:,l).'); % new one is column based
%             pp(l) = precomputed*fineshifts(:,l);
% 		end
		[mm,fineFreq_ind] = max(abs(pp));
		freqfound = fineFreq(fineFreq_ind);
        if (debugPlotsMode)
            figure(602+i); plot(fineFreq,abs(pp)); title('Fine Frequency Drift Check');
        end
    end
    if (debugPlotsMode)
        disp(['found FINE freq adjustment = ' num2str(freqfound)]);
    end
	finefreqfound = freqfound;
	
%     keyboard;
	% === Finer Time Resolution ===
%     x_aligned = x_aligned.*fineshifts(fineFreq_ind,:); % shift to correct frequency
    x_aligned = x_aligned.*fineshifts(:,fineFreq_ind).'; % shift to correct frequency

    x_fft = fft(x_aligned);
    y_fft = fft(y_aligned);
    rx_vec = (x_fft.*conj(y_fft));
    
%     sampleRes = 1/fs;
%     td_scan_range = (-5*sampleRes:100e-9:5*sampleRes); % we now pass it in via an argument
    
%     steeringvec = exp(1i*2*pi*freq.'*td_scan_range); % we now pass it in via an argument

%     cost_vec = (steeringvec*rx_vec.')/norm(x_fft)/norm(y_fft); % use this if you are using the mexIpp call to make steervec, it's better to transpose the 1-d vector than the huge matrix
%     cost_vec = (rx_vec*steeringvec); % this version is used in the realtime implementation currently

    cost_vec = (rx_vec*steeringvec)/norm(x_fft)/norm(y_fft); % this is for the above matlab original version, or the new freqContiguous mex call
    
    [QF,idx_td]=max(abs(cost_vec));
    timediff=td_scan_range(idx_td);
    if(debugPlotsMode)
        figure(601); clf(601); plot(td_scan_range,abs(cost_vec).^2,'b--'); hold on; plot(td_scan_range(idx_td),abs(cost_vec(idx_td)).^2,'ro');
        disp(['found FINE time adjustment = ' num2str(timediff)]);
        keyboard;
    end
    
end