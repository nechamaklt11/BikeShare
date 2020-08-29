function [tree_acc, tree_acc_avg, predictions_avg, tree_vperf] = tree(data,params)
%params: a structure with default setting
%vperf: average validation performance

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
for k=1:5
    tree=fitctree(train_inx{1,k}, targets_inx{1,k},...
        'MinLeafSize',params.MinLeafSize, 'MaxNumSplits', params.MaxNumSplits,  ...
        'PredictorNames', {'date', 'season', 'year', 'month', 'holiday', 'weekday', 'working day', 'situation', 'temp', 'atemp', 'humidity', 'windspeed'});
    
    [cost,secost,ntermnodes,bestlevel] = cvloss(tree, 'Subtrees','all');
    resubcost = resubLoss(tree, 'Subtrees','all');
    if params.plot==true
        plot_perform(ntermnodes, cost, resubcost, secost, bestlevel);
    end
    
    labelout(:,k)=predict(tree, tInputs);
    accuracy=0;
    for i=1:length(tTargets)
        if labelout(i,k)==tTargets(i)
            accuracy=accuracy+1;
        end
    end
    fold_acc(k)=accuracy/length(tTargets)*100;
    tree_vperf(k)=min(cost);
    
end
tree_vperf=mean(tree_vperf);

if params.plot==true
    if params.Prune==true
        pt = prune(tree,'Level',bestlevel);
        view(pt,'Mode','graph');
    else
        view(tree,'Mode','graph');
    end
    
    imp = predictorImportance(tree);
    figure;
    bar(imp);
    title('Predictor Importance Estimates');
    ylabel('Estimates');
    xlabel('Predictors');
    h = gca;
    h.XTickLabel = tree.PredictorNames;
    h.XTickLabelRotation = 45;
    h.TickLabelInterpreter = 'none';
end

%average cross validation performance
predictions_avg = round(mean(labelout'));
diff=predictions_avg-tTargets';
num_hits= sum(diff==0);
tree_acc_avg = (num_hits/length(test_targets))*100;

for k=1:5
    accuracy=0;
    for i=1:length(tTargets)
        if labelout(i,k)==tTargets(i)
            accuracy=accuracy+1;
        end
    end
    tree_acc(k)=accuracy/length(tTargets)*100;
end
tree_acc_avg=sum(tree_acc)/k;

function plot_perform(ntermnodes, cost, resubcost, secost, bestlevel)
%plot performance - training error vs validation error
    figure;
    plot(ntermnodes,cost,'b-', ntermnodes,resubcost,'r--')
    xlabel('Number of terminal nodes');
    ylabel('Cost (misclassification error)')
    [mincost,minloc] = min(cost);
    cutoff = mincost + secost(minloc);
    hold on
    plot([0 ntermnodes(1)], [cutoff cutoff], 'k:')
    plot(ntermnodes(bestlevel+1), cost(bestlevel+1), 'mo')
    legend('Cross-validation','Resubstitution','Min + 1 std. err.','Best choice')
    hold off


