function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

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


%   不带正则项
%   Theta1(25, 401); Theta2(0, 26)
X = [ones(m, 1), X];
z2 = Theta1 * X';
a2 = sigmoid(z2);
a2 = [ones(1, m); a2];
z3 = Theta2 * a2;
a3 = sigmoid(z3);
a_3 = 1 - a3;

Y = zeros(num_labels, m); %10*5000
for i = 1 : num_labels
    Y(i, y == i) = 1;
end

Y1 = 1 - Y;
res1 = 0;
res2 = 0;
for j = 1 : m
    %两个矩阵的每一列相乘,再把结果求和。预测值和结果label对应的元素相乘,就是某个输入x 的代价
    tmp1 = sum( log(a3(:,j)) .* Y(:,j) ); 
    res1 = res1 + tmp1; % m 列之和
    tmp2 = sum( log(a_3(:,j)) .* Y1(:,j) );
    res2 = res2 + tmp2;
end
J = (-res1 - res2) / m;

%   Forward propagation
for i = 1:m
    a1 = X(i, :)'; %the i th input variables, 400*1
    z2 = Theta1 * a1;
    a2 = sigmoid( z2 ); % Theta1 * x superscript i
    a2 = [1; a2 ];% add bias unit, a2's size is 26 * 1
    z3 = Theta2 * a2;
    a3 = sigmoid(z3); % h_theta(x)
    
    error_3 = a3 - Y( :, i); % last layer's error, 10*1
    %error_2 = ( Theta2' * error_3 ) .*  ( a2 .* (1 - a2) );% g'(z2)=g(z2)*(1-g(z2)), 26*1
    
    err_2 =  Theta2' * error_3; % 26*1
    error_2 = (err_2(2:end)) .* sigmoidGradient(z2);% 去掉 bias unit 对应的 error units
    
    Theta2_grad = Theta2_grad + error_3 * a2';
    Theta1_grad = Theta1_grad + error_2 * a1';
end

Theta2_grad = Theta2_grad / m; % video 9-2 backpropagation algorithm the 11 th minute
Theta1_grad = Theta1_grad / m;


%   加上 正则项

Theta1_tmp = Theta1(:, 2:end) .^ 2;
Theta2_tmp = Theta2(:, 2:end) .^ 2;
reg = lambda / (2*m) * (sum( Theta1_tmp(:) ) + sum(Theta2_tmp(:)));

J = (-res1 - res2) / m + reg;

Theta1(:,1) = 0;
Theta2(:,1) = 0;

Theta1_grad = Theta1_grad + lambda / m * Theta1;
Theta2_grad = Theta2_grad + lambda / m * Theta2;

grad = [Theta1_grad(:) ; Theta2_grad(:)];



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
