function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

Cs = [0.01 0.03 0.1 0.3 1 3 10 30];
sigmas = [0.01 0.03 0.1 0.3 1 3 10 30];

res = [];
for c=1:length(Cs)
  for s=1:length(sigmas)
    model = svmTrain(X, y, Cs(c), @(x1, x2) gaussianKernel(x1, x2, sigmas(s)));
    predictions = svmPredict(model, Xval);
    res = [res; mean(double(predictions ~= yval)) Cs(c) sigmas(s)];
  end
end

best = sortrows(res)(1,:);
C = best(2);
sigma = best(3);



% =========================================================================

end
