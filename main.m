%clear all; clc; close all;

bike_data = data_prep('day.csv') %read data and create training and test groups

visualize_data(bike_data) 
%%
nn_params=set_nn_params;
neural_network(inputs,targets,train_ind,valid_ind,nn_params);

