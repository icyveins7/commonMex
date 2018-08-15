function freq = makeFreq(len,fs)
    freq = [0:len-1-floor(len/2) -floor(len/2):-1]*fs/len;
end