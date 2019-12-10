% Encode json. Inputs are in cells. For lines, the x-y values should be
% saved in 2 columns (first column = x, second column = y).
function jsontext = geojsonEncode(points, pointlabels, lines, linelabels)
    if numel(points)~=numel(pointlabels)
        error('No. of points must equal no. of point labels.');
    end
    if numel(lines)~=numel(linelabels)
        error('No. of lines must equal no. of line labels.');
    end

    s.type = "FeatureCollection";
    s.features = {};
    
    % Add the points
    for pidx = 1:numel(points)
        f.type = "Feature";
        
        g.type = "Point";
        g.coordinates = points{pidx};
        
        f.geometry = g;
        
        p.label = pointlabels{pidx};
        
        f.properties = p;
        
        s.features{end+1} = f;
    end
    
    % Add the lines
    for lidx = 1:numel(lines)
        f.type = "Feature";
        
        g.type = "LineString";
        g.coordinates = lines{lidx};
        
        f.geometry = g;
        
        p.label = linelabels{lidx};
        
        f.properties = p;
        
        s.features{end+1} = f;
    end

    jsontext = jsonencode(s);
end