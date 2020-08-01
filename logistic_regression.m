function weights = logistic_regression(x,t,train_ind,valid_ind,params)
%EVERY COLUMN REPRESENTS A SAMPLE!!!
%params: max_epoch, threshold, learning_rate

lr = params.learning_rate;
max_epoch = params.max_epochs;
valid_errors = params.max_fail;
train_size = length(train_ind);

[num_features,num_samples]=size(x);
w =rand(num_features+1,1); %add one for bias
bias = ones(1,num_samples); %add bias after the last feature
x = [x ; bias]; %add bias row

rng(1); %for better randomization

%STOCHASTIC GRADIENT DESCENT
cost=zeros(1,max_epoch);
valid_cost = zeros(1,max_epoch);
epoch=1;
bad_valid = 0; %for early stop
best_w = w;

while epoch<max_epoch && bad_valid<valid_errors
    train_ind = randperm(train_size);
    for i=1:train_size
        sample_ind = train_ind(i);
        sample = x(:,sample_ind);
        target = t(sample_ind);
        wx = dot(sample,w); %multipluting the weights vector and the sample
        g = logistic_func(wx);
        w = w - lr*((g-target).*sample); %update weights
        cost(epoch)=cost(epoch)+cross_entropy(g,target);
    end
    cost(epoch)=cost(epoch)/train_size;
    %check validation
    X = x(:,valid_ind);
    T = t(valid_ind);
    valid_cost(epoch)= calc_valid_cost(X,T,w);
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

weights = best_w;
plot_perform(cost,valid_cost,epoch-1)

function g = logistic_func(x)
% logistic function
g=1./(1+exp(-1.*x));

function j = cross_entropy(g,t)
%compute cross entropy
j=t.*log(g)+(1-t).*log(1-g);
j=-j;

function valid_cost = calc_valid_cost(x,t,w)
%compute validation cost
y = w'*x;
g = logistic_func(y);
valid_cost=sum(cross_entropy(g,t))/length(t);

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






