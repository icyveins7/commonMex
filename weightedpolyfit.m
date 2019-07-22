function [X, STDX, MSE, S, mu, X_gradRidge] = weightedpolyfit(x,y,n,W,gradientRidgeLambda)
    x = x(:);
    y = y(:);

    % do the scaling and centre-ing like polyfit does
    mu = [mean(x); std(x)];
    x = (x - mu(1))/mu(2);
    
    % Construct the Vandermonde matrix V = [x.^n ... x.^2 x ones(size(x))]
    V(:,n+1) = ones(length(x),1,class(x));
    for j = n:-1:1
        V(:,j) = x.*V(:,j+1);
    end
    
    [Q,R] = qr(V,0);

    % now call lscov so we can use weights W
    [X, STDX, MSE, S_lscov] = lscov(V,y,W);
    
    r = y - V*X;
    % S is a structure containing three elements: the triangular factor
    % from a QR decomposition of the Vandermonde matrix, the degrees of
    % freedom and the norm of the residuals.
    S.R = R;
    S.df = max(0,length(y) - (n+1));
    S.normr = norm(r);
    
    % now we want to calculate an extra thing if the option is specified
    if (~isempty(gradientRidgeLambda))
        X_gradRidge = zeros(size(V,2), length(gradientRidgeLambda)); % initialize one new X for each lambda value
        
        min_t = min(V(:,end-1));
        max_t = max(V(:,end-1));
        
        g = [2*linspace(min_t, max_t, size(V,1)); ones(1, size(V,1)); zeros(1, size(V,1))].';
        
        for i = 1:length(gradientRidgeLambda)
            lambda = gradientRidgeLambda(i);
            X_gradRidge(:,i) = ((V.'*diag(W)*V) + lambda .* g.'*g) \ (V.'*diag(W)*y);
        end
    else
        X_gradRidge = [];
    end
end
    