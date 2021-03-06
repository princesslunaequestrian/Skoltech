function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_1_size, ...
                                   hidden_layer_2_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_1_size * (input_layer_size + 1)), ...
                 hidden_layer_1_size, (input_layer_size + 1));
             
start_Theta_2 = (hidden_layer_1_size * (input_layer_size + 1) + 1);
end_Theta_2 = start_Theta_2-1 + (hidden_layer_1_size+1)* hidden_layer_2_size;
             
Theta2 = reshape(nn_params(start_Theta_2:end_Theta_2), ...
                 hidden_layer_2_size, hidden_layer_1_size+1);  

Theta3 = reshape(nn_params(end_Theta_2 + 1:end), ...
                 num_labels, (hidden_layer_2_size + 1));
             
% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


X = [ones(m,1) X];


% foward propagation
% a1 = X; 

a2 = sigmoid(Theta1 * X');
a2 = [ones(m,1) a2'];
a3 = sigmoid(Theta2*(a2)');
a3 = [ones(size(a2,1),1) a3'];

h_theta = sigmoid(Theta3 * a3'); % h_theta equals z4

%one hot encoding
yk = zeros(num_labels, m); 
for i=1:m,
  yk(y(i),i)=1;
end

% follow the form
J = (1/m) * sum ( sum (  (-yk) .* log(h_theta)  -  (1-yk) .* log(1-h_theta) ));


% Note that you should not be regularizing the terms that correspond to the bias. 
% For the matrices Theta1 and Theta2,Theta 3 this corresponds to the first column of each matrix.
t1 = Theta1(:,2:size(Theta1,2));
t2 = Theta2(:,2:size(Theta2,2));
t3 = Theta3(:,2:size(Theta3,2));

% regularization formula
Reg = lambda  * (sum( sum ( t1.^ 2 )) + sum( sum ( t2.^ 2 )) + sum(sum(t3.^2))) / (2*m);

% cost function + reg
J = J + Reg;

% -------------------------------------------------------------

% Backprop

for t=1:m,

    % dummie pass-by-pass
    % forward propag

    a1 = X(t,:); % X already have bias
    z2 = Theta1 * a1';

    a2 = sigmoid(z2);
    a2 = [1 ; a2]; % add bias

    z3 = Theta2 * a2;

    a3 = sigmoid(z3);
    a3 = [1; a3]; % add bias
    
    z4 = Theta3 * a3; % final activation layer a4 == h(theta)
    a4 = sigmoid(z4);


    % back propag	

    z2=[1; z2]; % bias
    z3=[1; z3];
    delta_4 = a4 - yk(:,t);
    delta_3 = (Theta3' * delta_4) .* sigmoidGradient(z3);
    delta_3 = delta_3(2:end);
    delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z2);
    

    % skipping sigma2(0) 
    delta_2 = delta_2(2:end); 
    
    Theta3_grad = Theta3_grad + delta_4*a3';
    Theta2_grad = Theta2_grad + delta_3 * a2';
    Theta1_grad = Theta1_grad + delta_2 * a1; 

    end;

    % Regularization 


    Theta1_grad(:, 1) = Theta1_grad(:, 1) ./ m;

    Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) ./ m + ((lambda/m) * Theta1(:, 2:end));


    Theta2_grad(:, 1) = Theta2_grad(:, 1) ./ m;

    Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) ./ m + ((lambda/m) * Theta2(:, 2:end));
    
    Theta3_grad(:, 1) = Theta3_grad(:, 1) ./ m;

    Theta3_grad(:, 2:end) = Theta3_grad(:, 2:end) ./ m + ((lambda/m) * Theta3(:, 2:end));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:); Theta3_grad(:)];


end
