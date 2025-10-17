%% This code evaluates the test set.
clear all;
close all;
% ** Important.  This script requires that:
% 1)'centroid_labels' be established in the workspace
% AND
% 2)'centroids' be established in the workspace
% AND
% 3)'test' be established in the workspace

% IMPORTANT!!:
% You should save 1) and 2) in a file named 'classifierdata.mat' as part of
% your submission.
load('classifierdata.mat');
test = csvread('mnist_test_200_woutliers.csv');
correctlabels = test(:,785);
test = test(:,1:784);
test(:,785) = zeros(200,1);
predictions = zeros(200,1);
outliers = zeros(200,1);

% loop through the test set, figure out the predicted number
for i = 1:200

testing_vector= test(i,:);

% Extract the centroid that is closest to the test image
[prediction_index, vec_distance]=assign_vector_to_centroid(testing_vector,centroids);

predictions(i) = centroid_labels(prediction_index);
end

%% DESIGN AND IMPLEMENT A STRATEGY TO SET THE outliers VECTOR
% outliers(i) should be set to 1 if the i^th entry is an outlier
% otherwise, outliers(i) should be 0
% FILL IN
%calculating distance for all test samples
test_distances = zeros(200,1);
for i = 1:200
    testing_vector = test(i,1:784);
    [~, test_distances(i)] = assign_vector_to_centroid(testing_vector, centroids);
end
%outlier threshold(mean +2 standard deviation)
outlier_threshold = mean(test_distances) + 2*std(test_distances);
%outlier flag
for i = 1:200
    if test_distances(i) > outlier_threshold
        outliers(i) = 1;
    else
        outliers(i) = 0;
    end
end
%% MAKE A STEM PLOT OF THE OUTLIER FLAG
figure;
% FILL IN
stem(outliers, 'filled');
title('Outlier Detection Results');
xlabel('Test Sample Index');
ylabel('Outlier Flag (1=Outlier, 0=Normal)');
grid on;
%% The following plots the correct and incorrect predictions
% Make sure you understand how this plot is constructed
figure;
plot(1:200, correctlabels, 'bo', 'DisplayName', 'Correct Labels');
hold on;
plot(1:200, predictions, 'rx', 'DisplayName', 'Predictions');
% Mark outliers with black squares
outlier_indices = find(outliers == 1);
plot(outlier_indices, predictions(outlier_indices), 'ks', 'MarkerSize', 8, 'DisplayName', 'Outliers');
legend('Location', 'best');
title('Correct Labels vs Predictions');
xlabel('Test Sample Index');
ylabel('Digit Value');
grid on;

%% The following line provides the number of instances where and entry in correctlabel is
% equatl to the corresponding entry in prediction
% However, remember that some of these are outliers
sum(correctlabels==predictions)

function [index, vec_distance] = assign_vector_to_centroid(data,centroids)
% FILL IN
num_centroids = size(centroids, 1);
distances = zeros(num_centroids, 1);
pixel_data = data(1:784);
for j = 1:num_centroids
    current_centroid = centroids(j, 1:784);
    differences = pixel_data - current_centroid;
    squared_differences = differences .^ 2;
    sum_squared = sum(squared_differences);
    distances(j) = sqrt(sum_squared);
end

[vec_distance, index] = min(distances);
end


