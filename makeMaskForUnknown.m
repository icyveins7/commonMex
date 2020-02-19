% function for 0 offset starts with a burst
function mask =  makeMaskForUnknown(totallength, burstidxlen, guardidxlen, offsetidx)
    if offsetidx < 0
        error('Only positive or zero offset allowed.');
    end
    
    if offsetidx >=(burstidxlen+guardidxlen)
        error('Offset only up to total period - 1.');
    end
    mask = true(1,totallength);
    periodlen = burstidxlen + guardidxlen;
    % make the first one, assume a burst is the beginning
    if (offsetidx+1<=burstidxlen) % then still within a burst
        current_ptr = burstidxlen-offsetidx+1;
        mask(current_ptr:current_ptr+guardidxlen-1) = 0;
        current_ptr = current_ptr + guardidxlen;
    else % then within the guard
        current_ptr = periodlen - offsetidx;
        mask(1:current_ptr) = 0;
        current_ptr = current_ptr + 1;
    end
    
    remainlen = totallength - current_ptr + 1;
    
    while (remainlen > periodlen)
%         disp(['Start of loop, ' num2str(current_ptr)]);
        current_ptr = current_ptr + burstidxlen;
        
        mask(current_ptr:current_ptr+guardidxlen-1) = 0;
        current_ptr = current_ptr + guardidxlen;
%         disp(['End of loop, ' num2str(current_ptr)]);
        
        remainlen = totallength - current_ptr + 1;
    end % loop until one period (or less is left)
    

    
    if (remainlen > burstidxlen)
        current_ptr = current_ptr + burstidxlen;
        
        mask(current_ptr:end) = 0;
    end

end