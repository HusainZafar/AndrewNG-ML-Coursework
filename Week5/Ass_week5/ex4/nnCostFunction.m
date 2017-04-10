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

% Setup som e useful variables
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

 a1 = [ones(m,1),X];      #m=5000, a1=5000*401
 z2 = a1 * Theta1';       # (5000*401) * (25*401)'
 a2 = sigmoid(z2);        # 5000 * 25
 a2 = [ones(size(a2,1),1),a2]; # 5000 * 26
 z3 = a2 * Theta2';       # (5000 * 26) * (10 * 26)'
 a3 = sigmoid(z3);        # 5000 * 10
 
 y_matrix = eye(num_labels)(y,:) ;
 
 for i=1:m,
  for k=1:num_labels,
    J = J + (-y_matrix(i,k)*log(a3(i,k)) - (1-y_matrix(i,k))*log(1 - a3(i,k)));        
   end
 end

 J = J / m
 
  Theta1_reg = Theta1;
  Theta2_reg = Theta2;

  mat1 = Theta1_reg(:,[2:size(Theta1_reg,2)]).^2;
  mat2 = Theta2_reg(:,[2:size(Theta2_reg,2)]).^2;

  J = J + lambda/(2*m) * (sum(sum(mat1)) + sum(sum(mat2)))
 
% -------------------------------------------------------------

#vectorization
#m = no of ex, h = no of hidden layer, r=n of o/p, n = no oof training features includin bias 
d3 = a3 - y_matrix;     #del3 = m * r, z2 = m * h 
                            #Theta2 = r * (h-1), d3 = m * r
d2 = d3 * Theta2(:,2:end); #d2 = m * h
d2 = d2 .* sigmoidGradient(z2); #d2 = m * h

delta1 = d2' * a1;    #delta1 =  h * m  **   m * n = h * n
delta2 = d3' * a2;    #delta2 =  r * m  **  m * (h+1) = r * (h+1)
Theta1_grad = delta1 / m;
Theta2_grad = delta2 / m;
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m * Theta2(:,2:end);
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
