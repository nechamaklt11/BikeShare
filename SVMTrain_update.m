function SVMtrain = SVMTrain_update(inputs ,targets, test_inputs, test_targets, params)

inputsT=inputs'; targetsT=targets'; tInputs=test_inputs'; tTargets=test_targets';
rng default;
SVMmodel=fitcsvm(inputsT, targetsT, 'Solver', 'L1QP', ... %Using the quadric problem solving method
    'Standardize',true, 'KernelFunction','RBF',... 
    'KernelScale','auto', 'Verbose', 2, 'ClassNames',[0,1], ...
    'KFold', 5, 'Crossval', 'on'); % Cross-validation of 5 datasets from the training set

[labelout, score]=predict(SVMmodel.Trained{1,:}, tInputs);

%L1 = kfoldLoss(SVMmodel, 'mode','individual');
%L = loss(SVMmodel.Trained{1,:},tInputs,tTargets);

accuracy = 0;
for i=1:length(tTargets)
    if labelout(i)==tTargets(i)
        accuracy=accuracy+1;
    end
end
accuracy=accuracy/length(tTargets);
