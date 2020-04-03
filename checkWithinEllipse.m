% assumed that errorEllipse already repeats the first point.
% point is a column vector.
function check = checkWithinEllipse(errorEllipse, point)
    ybound = errorEllipse.lat_deg;
    xbound = errorEllipse.lon_deg;

    bound = [xbound; ybound];
    
    diff_vecs = bound - point;
%     raw_angles = zeros(1,size(diff_vecs,2) - 1);
    cross_angles = zeros(1,size(diff_vecs,2) - 1);
    
    for i = 1:size(diff_vecs,2) - 1
        vec0 = diff_vecs(:,i);
        vec1 = diff_vecs(:,i+1);
        
%         raw_angles(i) = acos(vec0.'*vec1 / norm(vec0) / norm(vec1));
        
        cross_pdt = cross([vec0;0], [vec1;0]);
        cross_angles(i) = cross_pdt(3);
    end
    
    
    check = all(cross_angles<0) || all(cross_angles>0);

%     keyboard;
end