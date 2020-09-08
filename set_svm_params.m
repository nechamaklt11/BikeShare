function params = set_svm_params()
params=struct('IterationLimit', 10000, 'Nu',0.05, 'BoxConstraint', 50, 'plot',true);
