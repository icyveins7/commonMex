function [ww, fv, aa] = czt_opt_prep(m, k, w, a, nfft)

% [m, n] = size(x); oldm = m;
% if m == 1, x = x(:); [m, n] = size(x); end

% if (m+k-1>nfft)
%     error('Need to pad fft points to greater than m+k-1!');
% end

% %------- Length for power-of-two fft.
% 
% nfft = 2^nextpow2(m+k-1);
% disp(['nfft = ' num2str(nfft)]);

%------- Premultiply data, note that all these are re-used!

kk = ( (-m+1):max(k-1,m-1) ).';
kk2 = (kk .^ 2) ./ 2;
% ww = w .^ (kk2);   % <----- Chirp filter is 1./ww
ww = exp(w .* kk2);   % <----- Chirp filter is 1./ww
% keyboard;

nn = (0:(m-1))';
% aa = a .^ ( -nn );
aa = exp(a .* -nn);
aa = aa.*ww(m+nn);

chirpfilter = 1 ./ ww(1:(k-1+m));
fv = fft( chirpfilter, nfft );   % <----- Chirp filter.

end