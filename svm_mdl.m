function [svm_acc, svm_acc_avg, predictions_avg, svm_vperf] = svm_mdl(data,params)

%INPUTS:
%data: inputs and targets, for test and train groups
%params: a structure with default setting
%OUTPUTS:
%svm_acc: accuracy for each fold
%avm_acc_avg: mean cross-validation accuracy
% predictions_avg: cross_validation model predictions
%svm_vperf: lowest validation performance, averaged across folds

load(data)
inputsT=inputs'; targetsT=targets'; tInputs=test_inputs'; tTargets=test_targets';

[train_ind,valid_ind]=cross_validation(length(targets),5); %train_ind, valid_ind: indeces for validation and train groups
for j=1:5
    for i=1:length(train_ind{1})
        train_inx{1,j}(i,:)=inputsT(train_ind{1,j}(1,i),:);
        targets_inx{1,j}(i,1)=targetsT(train_ind{1,j}(1,i),1);
    end
    for i=1:length(valid_ind{1})
        valid_inx{1,j}(i,:)=inputsT(valid_ind{1,j}(1,i),:);
        val_tar_inx{1,j}(i,1)=targetsT(valid_ind{1,j}(1,i),1);
    end
end

labelout=[]; loss_vec=[]; valid_vec=[];
for k=1:5
    n=1; 
    for i=5:5:length(train_inx{1,1})
        svm_mdl=fitcsvm(train_inx{1,k}([1:i],:), targets_inx{1,k}([1:i],:),'Solver','L1QP',...
            'PredictorNames', {'date', 'season', 'year', 'month', 'holiday', 'weekday', 'working day', 'situation', 'temp', 'atemp', 'humidity', 'windspeed'});
        loss_vec(n,k)=loss(svm_mdl, train_inx{1,k}, targets_inx{1,k}, 'LossFun', 'hinge');
        valid_vec(n,k)=loss(svm_mdl, valid_inx{1,k}, val_tar_inx{1,k}, 'LossFun', 'hinge');
        n=n+1; 
    end
    best_vperf(1,k)=min(valid_vec(:,k));
    labelout(:,k)=predict(svm_mdl, tInputs);
    accuracy=0;
    for i=1:length(tTargets)
        if labelout(i,k)==tTargets(i)
            accuracy=accuracy+1;
        end
    end
end
    
svm_vperf=mean(best_vperf);
    
predictions_avg = round(mean(labelout'));
diff=predictions_avg-tTargets';
num_hits= sum(diff==0);
svm_acc_avg = (num_hits/length(test_targets))*100;

for k=1:5
    accuracy=0;
    for i=1:length(tTargets)
        if labelout(i,k)==tTargets(i)
            accuracy=accuracy+1;
        end
    end
    svm_acc(k)=accuracy/length(tTargets)*100;
    
    if params.plot==true
        plot_perform(loss_vec(:,k), valid_vec(:,k),n-1); 
    end
end

function plot_perform(loss_vec,valid_vec,len)
%plot performance - training error vs validation error
figure
plot(1:len,loss_vec)
hold on
plot(1:len,valid_vec,'m')
xlim([0 len]);  ylim([0 3]);
title('SVM - Model Performance - Training Vs Validation ')
xlabel('Samples'); ylabel('Error (Hinge)')
legend('Training Error ','Validation Error')
hold off

