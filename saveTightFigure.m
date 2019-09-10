function saveTightFigure(h,outfilename)
% saves figure h in file outfilename without white space.
% 
% from some grad student bla bla bla, copied from old github code

% get current axes
ax = get(h,'CurrentAxes');
% make it tight
ti = get(ax,'TightInset');
set(ax,'Position',[1.15*ti(1) 1.1*ti(2) 0.98-ti(3)-ti(1) 1-ti(4)-ti(2)]);

% adjust the papersize
set(ax,'units','centimeters');
pos = get(ax,'Position');
ti = get(ax,'TightInset');
set(h, 'PaperUnits', 'centimeters');
set(h, 'PaperSize', [1.05*(pos(3)+ti(1)+ti(3)) 1.05*(pos(4)+ti(2)+ti(4))]);
set(h, 'PaperPositionMode', 'manual');
set(h, 'PaperPosition', [0 0 pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);

% save it
saveas(h,outfilename);