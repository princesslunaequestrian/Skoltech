clear; close all; clc;

Data = csvread('Real estate.csv');

Features = Data(2:end, 2:7); %Splitting Data into Features
I = Data(2:end, 1); %and Indexes
Info = Data(2:end, 8);

Features = [ones(length(Features), 1), Features]; %adding intercept

%Displaying fearures

for i=1:length(Features(1, :))
  %figure(i);
  %histfit(Features(:, i), 20);
end

train_index = int16(length(Features)*0.8); %data up to this index is training set
test_index = train_index + 1; %data from this index is test set


C = zeros(length(Features(1, :)), 1); %initial coefficient vector of all ones

iterations = 3000; 
alpha = 0.0000002; 

X = Features(1:train_index, :);
Y = Info(1:train_index, :);

%X = Features(test_index:end, :);
%Y = Info(test_index:end, :);

init_cost = Cost(X, Y, C);
fprintf('Initial cost: %e \n', init_cost);

[C, costs, Cs] = GradDesc(X, Y, C, alpha, iterations);

fprintf('Final cost for Training Dataset: %f \n', costs(end))
fprintf('Coefficients: \n');
disp(C);
fprintf('\n');

Test_X = Features(test_index:end, :);
Test_Y = Info(test_index:end, :);

fprintf('Test Cost: %f \n', Cost(Test_X, Test_Y, C));
fprintf('\n\n\n');

%normalize the normallest feature
%MinMax normalization

mn = min(Features(:, 6));
mx = max(Features(:, 6));

fprintf('Normalized set:\n\n');

X_norm = X;
X_norm(:, 6) = (X_norm(:, 6) - mn) / (mx - mn);
Y_norm = Y;

C2 = zeros(length(X_norm(1, :)), 1);

init_cost_norm = Cost(X_norm, Y_norm, C2);
fprintf('Initial cost for normalized: %f \n', init_cost_norm);

[C2, costs2, Cs2] = GradDesc(X_norm, Y_norm, C2, alpha, iterations);

fprintf('Final cost for  Training Normalized Dataset: %f \n', costs2(end))
fprintf('Coefficients 2: \n');
disp(C2);
fprintf('\n');


Test_X_norm = Test_X;
Test_X_norm(:, 6) = (Test_X_norm(:, 6) - mn) / (mx - mn);
Test_Y_norm = Test_Y;

fprintf('Normalized Test Cost: %f \n', Cost(Test_X_norm, Test_Y_norm, C2));


%Add non-linear features

X_nn = X;
Y_nn = Y;

X_nn = [X_nn, X_nn(:, 1).^2, X_nn(:, 1).*X_nn(:, 2)];

C_nn = zeros(length(X_nn(1, :)), 1);

fprintf('Initial cost for non-linear: %f, \n', Cost(X_nn, Y_nn, C_nn));

[C_nn, costs_nn, Cs_nn] = GradDesc(X_nn, Y_nn, C_nn, alpha, iterations);

fprintf('Final cost for Training Non-linear Dataset: %f \n', costs_nn(end))
fprintf('Coefficients_nn: \n');
disp(C_nn);
fprintf('\n');

Test_X_nn = Test_X;
Test_X_nn = [Test_X_nn, Test_X_nn(:, 1).^2, Test_X_nn(:, 1).*Test_X_nn(:, 2)];
Test_Y_nn = Test_Y;

fprintf('Non-linear Test Cost: %f \n', Cost(Test_X_nn, Test_Y_nn, C_nn));
