function [J, grad] = linearRegCostFunction(X, y, theta, lambda)

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

out = X*theta;
err = out - y;
sq = err.^2;
s = sum(sq)/(2*m);

reg = sum(theta'(1,2:end).^2) * lambda / (2 * m);
J = s + reg;

grad = (1/m) * sum((out - y).*X);
grad = grad + (lambda/m) * theta';
grad(1) = grad(1) - (lambda/m) * theta(1);

grad = grad(:);

end
