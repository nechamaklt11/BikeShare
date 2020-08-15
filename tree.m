function output = tree(inputs ,targets, test_inputs, test_targets)

inputsT=inputs'; targetsT=targets'; tInputs=test_inputs'; tTargets=test_targets';

t=fitctree(inputsT, targetsT, ...
    'OptimizeHyperparameters','all', ...
    'HyperparameterOptimizationOptions',struct('KFold', 5, 'SaveIntermediateResults', true, 'MaxObjectiveEvaluations', 20), ...
    'PredictorNames', {'date', 'season', 'year', 'month', 'holiday', 'weekday', 'working day', 'situation', 'temp', 'atemp', 'humidity', 'windspeed'});

resubcost = resubLoss(t ,'Subtrees','all');
[cost,secost,ntermnodes,bestlevel] = cvloss(t,'Subtrees','all');
figure(gcf);
plot(ntermnodes,cost,'b-', ntermnodes,resubcost,'r--')
xlabel('Number of terminal nodes');
ylabel('Cost (misclassification error)')
[mincost,minloc] = min(cost);
cutoff = mincost + secost(minloc);
hold on
plot([0 20], [cutoff cutoff], 'k:')
plot(ntermnodes(bestlevel+1), cost(bestlevel+1), 'mo')
legend('Cross-validation','Resubstitution','Min + 1 std. err.','Best choice')

pt = prune(t,'Level',bestlevel);
view(pt,'Mode','graph')

[labelout, score]=predict(t, tInputs);

accuracy = 0;
for i=1:length(tTargets)
    if labelout(i)==tTargets(i)
        accuracy=accuracy+1;
    end
end
accuracy=accuracy/length(tTargets)*100;
