clear; clc; close all;
%% DATA PROCESSING
bike_data_edited = data_prep('day_raw_data.csv'); %read data and create training and test groups
visualize_data(bike_data_edited)
load('data4analysis.mat')  %every column is a sample!!
[train_ind,valid_ind]=cross_validation(length(targets),5); 
 
%% NEURAL NETWORK
nn_params=set_nn_params;
[net_accuracy,net_accuracy_avg,predicted_net_y,net_vperf] =neural_network('data4analysis',nn_params);
accuracy_bar(net_accuracy, net_accuracy_avg, 'net accuracy across folds');

%% LOGISTIC REGRESSION 
reg_params = set_reg_params;
[reg_accuracy,reg_accuracy_avg,predicted_reg_y,reg_vperf] = logistic_regression('data4analysis',reg_params);
accuracy_bar(reg_accuracy, reg_accuracy_avg, 'logistic regression accuracy across folds');

%% DECISION TREES
tree_params = set_tree_params;
[tree_accuracy, tree_accuracy_avg, predicted_tree_y, tree_vperf]=tree('data4analysis',tree_params);
accuracy_bar(tree_accuracy, tree_accuracy_avg, 'trees accuracy across folds')

%% SVM model 
svm_params = set_svm_params;
[svm_accuracy, svm_accuracy_avg, predicted_svm_y, svm_vperf]=svm_mdl('data4analysis',svm_params);
accuracy_bar(svm_accuracy, svm_accuracy_avg, 'svm accuracy across folds')
%% GRID Search
% warning - extremely long running time
net_best_params=grid_search('learning_rate',[0 0.001 0.01 0.1 1],'max_epochs',...
    [1 10 100 200 1000],'hidden_layers',[2 1 3 5 10],'data4analysis',nn_params,1);
reg_best_params=grid_search('learning_rate',[0.001 0.01 0.1 0.5 1],'max_epochs',...
    [10 20 100 250,500],'lambda',[0 0.005 0.05 0.5],'data4analysis',reg_params,2);
tree_best_params=grid_search('MinLeafSize',[1 2 3 4 5], 'MaxNumSplits',[10 25 50 100 150],...
    'MaxNumCategories', [ 2 5 8 10 20], 'data4analysis',tree_params,3);
svm_best_params=grid_search('PassLimit', [1 5 10], 'LearnRate',[0.001 0.01 0.1 0.5 1] ,...
    'Lambda',[0.0005 0.005 0.05 0.5] ,'data4analysis',svm_params,4);
%% Ensemble Model
prediction = ensemble('data4analysis', nn_params, reg_params, tree_params, svm_params);
%% ENSEMBLE
predictions_all = [predicted_net_y;predicted_reg_y;predicted_tree_y;predicted_svm_y];
accuracy_all = [net_accuracy_avg,reg_accuracy_avg,tree_accuracy_avg,svm_accuracy_avg];
ensemble_acc = ensemble_func(accuracy_all,predictions_all,'data4analysis');
