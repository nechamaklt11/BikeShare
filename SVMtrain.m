function func = SVMtrain(inputs ,targets, params)
accuracy=[];
inputsT=inputs'; targetsT=targets';

%Cross validation
nr=length(targets');
bs=nr/5;
j=0;
for i=1:bs:nr
    v_set_x = inputsT(i:i+bs-1,1:11);
    v_set_y = targetsT(i:i+bs-1,1);
    if i==1
        t_set_x = inputsT(i+bs:nr,1:11); 
        t_set_y = targetsT(i+bs:nr,1);
    elseif i==nr-bs+1
        t_set_x = inputsT(1:i-1,1:11); 
        t_set_y = targetsT(1:i-1,1);
    else
        t_set_x = [inputsT(1:i-1,1:11) ; inputsT(i+bs:nr,1:11)]; 
        t_set_y = [targetsT(1:i-1,1) ; targetsT(i+bs:nr,1)];
    end
    
    trainedSVM = svmtrain(t_set_x ,t_set_y, ...
        'kernel_function', 'rbf', 'rbf_sigma', 1);
    
    outLabel = svmclassify(trainedSVM, v_set_x);
    
    %evaluate model
    acc=0;
    for i=1:length(outLabel)
        if outLabel(i)==v_set_y(i)
            acc=acc+1;
        end
    end
    
    j=j+1;
    accuracy(j)=acc/length(outLabel');
    
end

acc_mean = mean(accuracy)*100; %accuracy (precent)
acc_std = std(accuracy)*100; %std

%%