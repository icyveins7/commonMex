function [timetaken, testtext] = testasync
    tic;
    
    for i = 1:10
        fprintf('hihi \n');
        fprintf('this is iter %i\n',i);
    end
    testtext = 'what';
    
    timetaken = toc;
end