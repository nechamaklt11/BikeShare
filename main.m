%clear all; clc; close all;

bike_data = data_prep('day.csv'); %read data and create training and test groups

visualize_data(bike_data) 
%%
load('data_for_analysis') %data for testing
%data for analysis contains part of the bike data,divided to training and
%validation indeces
%every column is a sample!!!!!!!!!!!!!!!
nn_params=set_nn_params;
net =neural_network(inputs,targets,train_ind,valid_ind,nn_params);
% test net 


%%
reg_params = set_reg_params;
weights = logistic_regression(inputs,targets,train_ind,valid_ind,reg_params);


