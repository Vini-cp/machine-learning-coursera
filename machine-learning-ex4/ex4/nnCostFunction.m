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
a_2 = zeros(hidden_layer_size,1);
a_3 = zeros(num_labels,1);
delta_3 = zeros(num_labels,m); % 10 x 5000  Each column reprents an input layer error (10 outputs = 10 errors/input)
delta_2 = zeros(hidden_layer_size,m); % 25 x 5000 Each column reprents an input layer error (25 outputs = 25 errors/input)

% size(theta_1) 	= 25 x 401
% size(theta_2) 	= 10 x 26

% size(X(i,:)) 		= 1 x 401 (Each line is a digit example)
% size(Theta1*X')' 	= 1 x 25
% size(Theta2*X')'	= 1 x 10

% size(y) 			= 1 x 1
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
add_column_1 = zeros(num_labels,1);
add_column_2 = zeros(hidden_layer_size,1);

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


X = [ones(size(X,1),1) X];

i = 0;
k = 0;
w = 0;

Teste1 = sum((Theta1.^2));
Cost_Theta1 = sum(Teste1(2:size(Teste1,2)));
Cost_Theta1;

Teste2 = sum((Theta2.^2));
Cost_Theta2 = sum(Teste2(2:size(Teste2,2)));
Cost_Theta2;

for i = 1:m 																% Backpropagation - Step 1
	a_2 = sigmoid(Theta1*X(i,:)')';
	a_2 = [1 a_2];
	a_3 = sigmoid(Theta2*a_2')';
	
	for k = 1:num_labels
		if y(i) == k
			w = 1;
		else
			w = 0;
		end

		J = J - (1/m)*(w*log(a_3(k)) + (1-w)*log(1-a_3(k)));				% Regularized cost function

		delta_3(k,i) = a_3(k)-w;											% Backpropagation - Step 2
	end

																			% Backpropagation - Step 3
	
	delta_2(:,i) = Theta2(:,2:end)'*delta_3(:,i).*sigmoidGradient(Theta1*X(i,:)');

	Theta2_grad = Theta2_grad + delta_3(:,i)*a_2;	% Backpropagation - Step 4

	Theta1_grad = Theta1_grad + delta_2(:,i)*X(i,:);

end

J = J + (Cost_Theta1 + Cost_Theta2)*(lambda/(2*m));


% -------------------------------------------------------------

Theta1_grad = (1/m).*Theta1_grad + [zeros(size(Theta1,1),1) (lambda/m)*Theta1(:,2:end)];
Theta2_grad = (1/m).*Theta2_grad + [zeros(size(Theta2,1),1) (lambda/m)*Theta2(:,2:end)];



% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
