function prediction = ensemble (data, params1, params2, params3, params4)
lr_acc=0; tree_acc=0; svm_acc=0;
load(data)

[~,~,predicted_y(:,1),~] = neural_network(data,params1);
[~,~,predicted_y(:,2),~] = logistic_regression(data,params2);
[~,~,predicted_y(:,3),~] = tree(data,params3);
[~,~,predicted_y(:,4),~] = svm_mdl(data,params4);
for i=1:length(predicted_y)
    predicted_y(i,5)=sum(predicted_y(i,[2:4]));
end

for i=1:length(predicted_y)
    if predicted_y(i,5)>=2
        predicted_y(i,5)=1
    else
        predicted_y(i,5)=0
    end
end

for i=1:length(predicted_y)
    if predicted_y(i,5)== predicted_y(i,2)
        lr_acc=lr_acc+1;
    end
    if predicted_y(i,5)== predicted_y(i,3)
        tree_acc=tree_acc+1;
    end
    if predicted_y(i,5)== predicted_y(i,4)
        svm_acc=svm_acc+1;
    end
end
lr_acc=lr_acc/length(predicted_y)*100;
tree_acc=tree_acc/length(predicted_y)*100;
svm_acc=svm_acc/length(predicted_y)*100;

acc=[lr_acc tree_acc svm_acc];
bar(acc, 'b');
set(gca, 'xticklabels',({'Logistic regression','Decision Trees','SVM'}));
title('Accuracy of each model for test data set');
