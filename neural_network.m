
function neural_network(x,t,train_ind,valid_ind,params) 
%applying a neural network on input x and target t.
%train_ind,valid_ind: indeces for validation and train groups
%params: a structure with default setting

rng(1) %for better randomization

%CREATE NETWORK
net = feedforwardnet(params.hidden_layers,'traingd');  %gradient decent training 
net.numInputs=1; %we have one input with a few parameters

% SET PARAMETERS & HYPERPARAMETERS
net.layers{1}.transferFcn='logsig'; %training function is sigmoid
net.layers{2}.transferFcn='tansig'; %output layer. can be either tansig or pure linear
net.performFcn ='crossentropy'; %matching error function - X entropy
net.trainParam.epochs = params.max_epochs; 
net.trainParam.lr = params.learning_rate;
net.trainParam.max_fail = params.max_fail; %early stop condition 
net.performParam.regularization =params.lambda;
net.divideFcn = 'divideind'; %divide train and validation groups manually
net.divideParam.trainInd =train_ind; net.divideParam.valInd  = valid_ind; net.divideParam.testInd  = []; %use default validation and training indeces
net.trainParam.showWindow = false;

%TRAIN NETWTORK AND COMPARE TO VALIDATION GROUP
[net,tr]=train(net,x,t); %training the network
plotperform(tr) %plotting performance graph
view(net) %visualize network activity
end
