function params = set_nn_params
%creating a struct with parameters for neural network and setting default values.
params=struct('learning_rate',0.1,'max_epochs',1000,'hidden_layers',10,'lambda',0.01,'max_fail',5,'plot',false);