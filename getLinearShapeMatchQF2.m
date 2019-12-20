% rehashed from the processMessagesRT version, for a more general input
% note that in this version, we compare against qf2/(1-qf2) for a more
% constant linear coefficient shape match
% QF2 expected as column vector, shape2match expected as column vector.
function shapeqf2 = getLinearShapeMatchQF2(shape2match, qf2maxIdx, qf2)
    shapelen = length(shape2match); % this is assumed to be even
    shapeqf2 = 1.0; % default to this in case of failures, or if not long enough data to match
    
%     linearqf2 = qf2./(1-qf2); % disabled for now
    linearqf2 = qf2;
    
    if (qf2maxIdx > shapelen/2 && qf2maxIdx < length(qf2) - shapelen/2) % only process it if there's enough elements to do the dot product, otherwise output default value
        testpattern = linearqf2(qf2maxIdx - shapelen/2 : qf2maxIdx + shapelen/2 - 1);
        shapeqf2 = (testpattern.' * shape2match)^2 / norm(testpattern)^2 / norm(shape2match)^2;  
    else
        fprintf('Peak sides not enough elements, defaulting to 1.0\n');
    end
%     keyboard;
end