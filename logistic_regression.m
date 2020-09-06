function [accuracy,accuracy_avg,predictions_avg,vperf] = logistic_regression(data,params)
%EVERY COLUMN REPRESENTS A SAMPLE!!!
%params: max_epoch, threshold, learning_rate
load(data)
x=inputs;
t=targets;
[train_idx,valid_idx]=cross_validation(length(targets),5); %train_ind,valid_ind: indeces for validation and train groups

lr = params.learning_rate;
max_epoch = params.max_epochs;
valid_errors = params.max_fail;
lambda = params.lambda;
train_size = length(train_idx{1});

[num_features,num_samples]=size(x);
bias = ones(1,num_samples); %add bias after the last feature
x = [x ; bias]; %add bias row
accuracy = zeros(1,5);
predictions = zeros(5,length(test_targets)) ;
best_vperf=zeros(1,5); %best validation performance
rng(1) %for better randomization

for k=1:5
    w =rand(num_features+1,1); %add one for bias
    train_ind=train_idx{k};
    valid_ind=valid_idx{k};
    %STOCHASTIC GRADIENT DESCENT
    cost=zeros(1,max_epoch);
    valid_cost = zeros(1,max_epoch);
    epoch=1;
    bad_valid = 0; %for early stop
    best_w = w;
    
    while epoch<max_epoch && bad_valid<valid_errors
        Ind = randperm(train_size);
        for i=1:train_size
            sample_ind = train_ind(Ind(i));
            sample = x(:,sample_ind);
            target = t(sample_ind);
            wx = dot(sample,w); %multipluting the weights vector and the sample
            g = logistic_func(wx);
            w = w - lr*((g-target).*sample+(2*lambda).*w); %update weights
            cost(epoch)=cost(epoch)+cross_entropy(g,target)+lambda*sum(w.^2);
        end
        cost(epoch)=cost(epoch)/train_size;
        %check validation
        X = x(:,valid_ind);
        T = t(valid_ind);
        valid_cost(epoch)= calc_valid_cost(X,T,w,lambda);
        if epoch>2 && valid_cost(epoch)>valid_cost(epoch-1)
            bad_valid = bad_valid+1;
        else
            bad_valid = 0;
            best_w=w;
        end
        epoch = epoch+1;
    end
    
    cost = cost(1:epoch-1);
    valid_cost = valid_cost(1:epoch-1);
    best_vperf(1,k)=min(valid_cost);
    
    %test accuracy
    X = [test_inputs; ones(1,size(test_inputs,2))]; %add a bias row
    [accuracy(k),predictions(k,:)] = test_regression(X,test_targets,best_w);
    if params.plot==true
        plot_perform(cost,valid_cost,epoch-1)
    end
end

%average cross validation performance
predictions_avg = round(mean(predictions,1));
diff=predictions_avg-test_targets;
num_hits= sum(diff==0);
accuracy_avg = (num_hits/length(test_targets))*100;
vperf=mean(best_vperf);

function g = logistic_func(x)
% logistic function
g=1./(1+exp(-1.*x));

function j = cross_entropy(g,t)
%compute cross entropy
j=t.*log(g)+(1-t).*log(1-g);
j=-j;

function valid_cost = calc_valid_cost(x,t,w,lambda)
%compute validation cost
y = w'*x;
g = logistic_func(y);
valid_cost=sum(cross_entropy(g,t)+lambda*sum(w.^2))/length(t);

function [accuracy,predictions] = test_regression(x,t,w)
y = w'*x;
y=logistic_func(y);
predictions(y<0.5)=0; predictions(y>=0.5)=1;
diff=predictions-t;
num_hits= sum(diff==0);
accuracy = (num_hits/length(t))*100;

function plot_perform(cost,valid_cost,epoch)
%plot performance - training error vs validation error
figure
plot(1:epoch,cost)
hold on
plot(1:epoch,valid_cost,'m')
title('Linear Regression - Model Performance - Training Vs Validation ')
xlabel('Epoch'); ylabel('Error (Cross-Entropy)')
legend('Training Error ','Validation Error')
hold off
