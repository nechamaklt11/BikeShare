clear; clc; close all;

%% DATA PROCESSING
bike_data_edited = data_prep('day.csv'); %read data and create training and test groups
visualize_data(bike_data_edited)
load('data4analysis')  %every column is a sample!!
[train_ind,valid_ind]=cross_validation(length(targets),5); 
 
%% NEURAL NETWORK
nn_params=set_nn_params;
[net_accuracy,net_accuracy_avg,predicted_net_y,net_vperf] =neural_network('data4analysis',nn_params);
accuracy_bar(net_accuracy, net_accuracy_avg, 'net accuracy across folds')

%% LOGISTIC REGRESSION 
reg_params = set_reg_params;
[reg_accuracy,reg_accuracy_avg,predicted_reg_y,reg_vperf] = logistic_regression('data4analysis',reg_params);
accuracy_bar(reg_accuracy, reg_accuracy_avg, 'logistic regression accuracy across folds')

%% SVM model 
SVMTrain_update(inputs ,targets, test_inputs, test_targets);

%% Tree model 
tree(inputs ,targets, test_inputs, test_targets);

%%
net_best_params=grid_search('learning_rate',[0 0.001 0.01 0.1 1],'max_epochs',[1 10 100 200 1000],'hidden_layers',[2 1 3 5 10],'data4analysis',nn_params,1);
%logistic_best_params=grid_search()