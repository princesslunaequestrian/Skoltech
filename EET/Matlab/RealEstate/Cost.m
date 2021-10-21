function cost = Cost(X, Y, C)
  
  m = length(Y);
  
  cost = 1.0 / (2.0 * m) * ((X*C - Y)' * (X*C - Y));
  
end