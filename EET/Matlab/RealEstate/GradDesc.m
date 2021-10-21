function [C, Costs, Cs] = GradDesc(X, Y, C, step, iters)
  
  m = length(Y);
  Costs = zeros(iters, 1);
  Cs = zeros(iters, length(X(1, :)));
  
  
  for i = 1:iters
    
    tmp = C;
    
    C = tmp - step * ((X') * (X * tmp - Y))/m;
    
    Costs(i) = Cost(X, Y, C);
    Cs(i, :) = C;
    
  endfor
end