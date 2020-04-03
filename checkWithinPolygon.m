function check = checkWithinPolygon(polygon, pt)
    ybound = polygon.lat_deg;
    xbound = polygon.lon_deg;

    bound = [xbound; ybound];
    
    diff_vecs = bound - pt;

    dd = 0;
    for k = 1:size(diff_vecs,2) - 1
        avec = diff_vecs(:,k);
        bvec = diff_vecs(:,k+1);
        da = atan2(avec(2), avec(1));
        db = atan2(bvec(2), bvec(1));
%         disp(db-da)
        dc = db-da;
        if (dc<-pi)
            dc = dc + 2 * pi;
        end
        if (dc>=pi)
            dc = dc - 2*pi;
        end
        dd = dd + dc;
        
    end

    check = dd > 2*pi - 0.1;
    
%     keyboard;
end