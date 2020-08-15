function SVMtrain = SVMTrain_update(inputs ,targets, test_inputs, test_targets)

inputsT=inputs'; targetsT=targets'; tInputs=test_inputs'; tTargets=test_targets';
rng default;
SVMmodel=fitcsvm(inputsT, targetsT,  ... %Using the quadric problem solving method
    'OptimizeHyperparameters','auto', ...
    'HyperparameterOptimizationOptions',struct('SaveIntermediateResults', true, 'MaxObjectiveEvaluations', 10)); 

cvmodel = crossval(SVMmodel,'KFold',5);

[labelout, score]=predict(cvmodel.Trained{1,:}, tInputs);

accuracy = 0;
for i=1:length(tTargets)
    if labelout(i)==tTargets(i)
        accuracy=accuracy+1;
    end
end

accuracy=accuracy/length(tTargets)*100;
