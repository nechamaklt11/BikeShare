clear; clc; close all;

%% DATA PROCESSING
bike_data_edited = data_prep('day.csv'); %read data and create training and test groups
visualize_data(bike_data_edited)
load('data4analysis')
 %every column is a sample!!

 %%
 [train_ind,valid_ind]=cross_validation(length(targets),5);
 
%% NEURAL NETWORK
nn_params=set_nn_params;
nn_params.lr=0.01;
nn_params.hidden_layers=10;
nn_params.lambda=0.01;
net_accuracy=zeros(1,5);
predicted_y=cell(1,5);
for i=1:5
net =neural_network(inputs,targets,train_ind{i},valid_ind{i},nn_params);
predicted_y{i}=net(test_inputs);
net_accuracy(i) = evaluate_net(test_targets,predicted_y{i});
end
view(net)

%% LOGISTIC REGRESSION
reg_params = set_reg_params;
reg_params.lambda=0.005;
reg_params.lr=0;
reg_params.max_epochs=200;
accuracy = zeros(1,5);
for i=1:5
weights = logistic_regression(inputs,targets,train_ind{i},valid_ind{i},reg_params);
X = [test_inputs; ones(1,size(test_inputs,2))]; %add a bias row
accuracy(i) = test_regression(X,test_targets,weights);
end

