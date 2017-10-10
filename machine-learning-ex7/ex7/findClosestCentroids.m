function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

temp_list = zeros(size(X,1), 1);
for i = 1 : length(temp_list)
    temp_list(i) = 10000;
end
%disp(temp_list);
for i = 1 : length(idx)
    for j = 1 : K
        temp_num = (X(i, :) - centroids(j, :)) * (X(i, :) - centroids(j, :))';
        if temp_num <= temp_list(i)
            temp_list(i) = temp_num;
            idx(i) = j;
        end
    end    
end    



% =============================================================

end

