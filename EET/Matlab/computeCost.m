function J = computeCost(X, y, theta)
  m = length(y); % number of training examples
  J = 1.0 / (2.0 * m) * (X * theta - y)' * (X * theta - y);
end
