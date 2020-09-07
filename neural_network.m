function [accuracy,accuracy_avg,predicted_y_avg,vperf] = neural_network(data,params)
%applying a neural network on input x and target t.
%INPUTS:
%data: inputs and targets, for test and train groups
%params: a structure with default setting
%OUTPUTS:
%accuracy: accuracy for each fold
%accuracy_avg: mean cross-validation accuracy
% predicted_y_avg: cross_validation model predictions
%vperf: lowest validation performance, averaged across folds
load(data)
x=inputs;
t=targets;
[train_ind,valid_ind]=cross_validation(length(targets),5); %train_ind,valid_ind: indeces for validation and train groups

rng(2) %for better randomization

predicted_y=zeros(5,length(test_targets));
accuracy=zeros(1,5);
best_vperf=zeros(1,5);
for i=1:5
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
    net.trainParam.showWindow = false;
    net.divideParam.trainInd =train_ind{i}; net.divideParam.valInd  = valid_ind{i}; net.divideParam.testInd  = []; %use default validation and training indeces
    
    %TRAIN NETWTORK AND COMPARE TO VALIDATION GROUP
    [net,tr]=train(net,x,t); %training the network
    if params.plot==true
        plotperform(tr) %plotting performance graph
        view(net)
    end
    predicted_y(i,:)=net(test_inputs);
    accuracy(i)= evaluate_net(test_targets,predicted_y(i,:));
    best_vperf(i) = tr.best_vperf;
end
predicted_y_avg=round(mean(predicted_y,1));
accuracy_avg=evaluate_net(test_targets,predicted_y_avg);
vperf = mean(best_vperf);

function accuracy = evaluate_net(targets,predictions)
predictions(predictions<0.5)=0; predictions(predictions>=0.5)=1;
diff = targets-predictions;
num_hits= sum(diff==0);
accuracy = (num_hits/length(targets))*100;

