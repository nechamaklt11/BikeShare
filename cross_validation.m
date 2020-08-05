function [train_ind,valid_ind]=cross_validation(N,K)
%N=number of observations
%k=number of folds
%the outputs are cells
C= cvpartition(N,'KFold',K);
train_ind=cell(1,5);
valid_ind=cell(1,5);
for i=1:K
    train_ind{i}=find(C.training(i)==1);
    valid_ind{i}=find(C.test(i)==1);
end

    