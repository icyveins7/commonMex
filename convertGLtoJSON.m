% Cells should be filled with {variable of lons, variable of lats, label, variable
% of lons, variable of lats, label ...}
function encoded = convertGLtoJSON(gl_cell, set_label, jsonfilename)
    s_list = {};
    for i = 0:length(gl_cell)/3 - 1
        lon = gl_cell{i*3+1};
        lon = lon(:);
        lat = gl_cell{i*3+2};
        lat = lat(:);
        
        full = [lon lat];
        if length(lon)>1
            s = struct('type','line','coords',full,'label',gl_cell{i*3+3});
        else
            s = struct('type','pt','coords',full,'label',gl_cell{i*3+3});
        end
        s_list{end+1} = s;
    end
    
    s_json.features = s_list;
    s_json.set_label = set_label;

    encoded = jsonencode(s_json);
    
    fid = fopen(jsonfilename,'w');
    fprintf(fid,'%s',encoded);
    fclose(fid);
end