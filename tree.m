function [tree_acc, tree_acc_avg, predictions_avg, tree_vperf] = tree(inputs ,targets, test_inputs, test_targets, data)

load(data)
x=inputs';
t=targets';
[train_idx,valid_idx]=cross_validation(length(targets),5); %train_ind,valid_ind: indeces for validation and train groups
train_size = length(train_idx{1});

A{1,1}=x([1:117],:);
A{1,2}=x([118:234],:);
A{1,3}=x([234:350],:);
A{1,4}=x([351:467],:);
A{1,5}=x([469:585],:);
B{1,1}=t([1:117]);
B{1,2}=t([118:234]);
B{1,3}=t([234:350]);
B{1,4}=t([351:467],:);
B{1,5}=t([469:585],:);

inputsT=inputs'; targetsT=targets'; tInputs=test_inputs'; tTargets=test_targets';

weights=ones(117,1);
labelout=[];

for k=1:5
    tree=fitctree(A{1,k}, B{1,k}, ...
        'MinLeafSize',2, 'MaxNumSplits', 50, 'Prune', 'on', 'Weights', weights,...
        'PredictorNames', {'date', 'season', 'year', 'month', 'holiday', 'weekday', 'working day', 'situation', 'temp', 'atemp', 'humidity', 'windspeed'});
    
    [cost,secost,ntermnodes,bestlevel] = cvloss(tree, 'Subtrees','all');
    resubcost = resubLoss(tree, 'Subtrees','all');
    plot_perform(ntermnodes, cost, resubcost, secost, bestlevel);
    
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
%'MinLeafSize',2, 'MaxNumSplits', Inf, 'Prune', 'off', 'Weights', weights,...
tree_vperf=mean(tree_vperf);
pt = prune(tree,'Level',bestlevel);
view(pt,'Mode','graph');

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

n=1;
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
end

end