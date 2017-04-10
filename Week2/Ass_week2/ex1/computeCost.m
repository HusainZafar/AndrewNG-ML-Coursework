function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


%X will be a mx2 matrix and theta is 2x1. X*theta will give the hypothesis function given as htheta(x)=x1+theta*x2

hyp=X*theta;
s=(hyp-y);
squarederr=s.^2;
summation=sum(squarederr);
J=summation/(2*m)



% =========================================================================

end
