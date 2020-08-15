clear; clc; close all;

%% DATA PROCESSING
bike_data_edited = data_prep('day.csv'); %read data and create training and test groups
visualize_data(bike_data_edited)
load('data4analysis')  %every column is a sample!!
[train_ind,valid_ind]=cross_validation(length(targets),5); 
 
%% NEURAL NETWORK
nn_params=set_nn_params;
[net_accuracy,net_accuracy_avg]=neural_network(inputs,targets,train_ind,valid_ind,nn_params,test_inputs,test_targets);

%% LOGISTIC REGRESSION
reg_params = set_reg_params;
[reg_accuracy,reg_accuracy_avg] = logistic_regression(inputs,targets,train_ind,valid_ind,reg_params,test_inputs,test_targets);

%% SVM model 
SVMTrain_update(inputs ,targets, test_inputs, test_targets);

%% Tree model 
tree(inputs ,targets, test_inputs, test_targets);
