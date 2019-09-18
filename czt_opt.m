function g = czt_opt(x, k, premul_ww, premul_aa, nfft, premul_fv)


[m, n] = size(x); oldm = m;
if m == 1, x = x(:); [m, n] = size(x); end

% if (m+k-1>nfft)
%     error('Need to pad fft points to greater than m+k-1!');
% end

% %------- Length for power-of-two fft.
% 
% nfft = 2^nextpow2(m+k-1);
% disp(['nfft = ' num2str(nfft)]);

%------- Premultiply data, note that all these are re-used!

% kk = ( (-m+1):max(k-1,m-1) ).';
% kk2 = (kk .^ 2) ./ 2;
% ww = w .^ (kk2);   % <----- Chirp filter is 1./ww
% ww = exp(w .* kk2);   % <----- Chirp filter is 1./ww, using phase info
% nn = (0:(m-1))';
% % aa = a .^ ( -nn );
% aa = exp(a .* -nn);
% % aa = aa.*ww(m+nn);
% aa = aa.*premul_ww(m+nn); % use the prepared ww instead
% y = x .* aa(:,ones(1,n));
y = x .* premul_aa;

%------- Fast convolution via FFT.

fy = fft(  y, nfft );
% fv = fft( 1 ./ ww(1:(k-1+m)), nfft );   % <----- Chirp filter.
% fy = fy .* fv(:,ones(1, n));
fy = fy .* premul_fv; % use prepared fv instead
g  = ifft( fy );

%------- Final multiply.

g = g( m:(m+k-1), : ) .* premul_ww( m:(m+k-1),ones(1, n) );

if oldm == 1, g = g.'; end

