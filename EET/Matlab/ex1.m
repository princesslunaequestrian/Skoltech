%% Initialization
clear ; close all; clc

%% ======================= Plotting =======================
fprintf('Plotting Data ...\n')
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2); Z = X.*X;
m = length(y); % number of training examples

% Plot Data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =================== Gradient descent ===================
fprintf('Running Gradient Descent ...\n')

X = [ones(m, 1), X, Z]; % Add a column of ones and X^2 to x
theta = zeros(3, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

% compute and display initial cost
J = computeCost(X, y, theta);
fprintf('Initial cost %f \n', J)

% run gradient descent
[theta, J_history] = gradientDescent(X, y, theta, alpha, iterations);

% compute and display final cost
J = computeCost(X, y, theta);
fprintf('Final cost %f \n', J)

% print theta to screen
fprintf('Theta found by gradient descent: ');
fprintf('%f %f \n', theta(1), theta(2));

% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

figure;
plot(1:iterations, J_history)

% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5, 1] *theta;
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7, 1] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============= Part 4: Visualizing J(theta_0, theta_1) =============
fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);
theta2_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals), length(theta2_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
      for k = 1:length(theta2_vals)
	      t = [theta0_vals(i); theta1_vals(j); theta2_vals(k)];    
	      J_vals(i,j, k) = computeCost(X, y, t);
      end
    end
end


% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);