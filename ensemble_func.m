function accuracy = ensemble_func(accuracy_all,predictions_all,data)
%INPUTS:
% accuracy_all: accuracy for each model.
%predictions_all: test_predictions for each model. 
%data: test and training groups data.
%OUTPUS:
%accuracy:ensemble model accuracy

load(data)
avg_p = round(mean(predictions_all,1));
num_hits=sum(test_targets==avg_p);
accuracy = (num_hits/length(avg_p))*100;

bar([accuracy_all,accuracy])
set(gca, 'xticklabels',({'Neural Network','Logistic regression','Decision Trees','SVM','Ensemble'}));
title('Models Accuracy Comparison');
ylim([70 100])









