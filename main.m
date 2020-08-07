clear; clc; close all;

%% DATA PROCESSING
bike_data = data_prep('day.csv'); %read data and create training and test groups
visualize_data(bike_data) 
load('data_for_analysis.mat') %every column is a sample!!!!!!!!!!!!!!!

%% NEURAL NETWORK
nn_params=set_nn_params;
net=neural_network(inputs,targets,train_ind,valid_ind,nn_params);
view(net);
predicted_y=net(test_inputs);
net_accuracy = evaluate_net(test_targets,predicted_y);

%% LOGISTIC REGRESSION
reg_params = set_reg_params;
weights = logistic_regression(inputs,targets,train_ind,valid_ind,reg_params);
X = [test_inputs; ones(1,size(test_inputs,2))]; %add a bias row
accuracy = test_regression(X,test_targets,weights);

%% SVM model 
SVMtrain(inputs ,targets, nn_params);
