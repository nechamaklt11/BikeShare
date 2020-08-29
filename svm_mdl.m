function [svm_acc, svm_acc_avg, predictions_avg, svm_vperf] = svm_mdl(data,params)

load(data)
inputsT=inputs'; targetsT=targets'; tInputs=test_inputs'; tTargets=test_targets';

[train_ind,valid_ind]=cross_validation(length(targets),5); %train_ind, valid_ind: indeces for validation and train groups
for j=1:5
    for i=1:length(train_ind{1})
        train_inx{1,j}(train_ind{1,j}(1,i),:)=inputsT(train_ind{1,j}(1,i),:);
        targets_inx{1,j}(train_ind{1,j}(1,i),1)=targetsT(train_ind{1,j}(1,i),1);
    end
end

labelout=[];
svm_mdl=fitclinear(inputsT, targetsT,'CrossVal','on','Kfold',5, ...
    'Solver', 'dual', 'IterationLimit', params.IterationLimit, ...
    'LearnRate', params.LearnRate, 'Lambda', params.Lambda,...
    'PredictorNames', {'date', 'season', 'year', 'month', 'holiday', 'weekday', 'working day', 'situation', 'temp', 'atemp', 'humidity', 'windspeed'});
    
L = kfoldLoss(svm_mdl, 'lossfun', 'classiferror', 'mode', 'individual');
svm_vperf=min(L);

for k=1:5
    labelout(:,k)=predict(svm_mdl.Trained{k,1}, tInputs);
    accuracy=0;
    for i=1:length(tTargets)
        if labelout(i,k)==tTargets(i)
            accuracy=accuracy+1;
        end
    end
    fold_acc(k)=accuracy/length(tTargets)*100;
end
    
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
end
svm_acc_avg=sum(svm_acc)/k;

if params.plot==true
   n=1; 
end