function optimal_vals = grid_search(param1,val1,param2,val2,param3,val3,data,params,mode)

load(data)
combs=combvec(val1,val2,val3); %all possible combinations

%initialize best validation performance and optimal values
best_vperf=100;
optimal_vals=[0,0,0];

for i=1:length(combs)
    p1 = combs(1,i);
    p2=combs(2,i);
    p3=combs(3,i);
    params=setfield(params,param1,p1);
    params=setfield(params,param2,p2);
    params=setfield(params,param3,p3);
    switch mode
        case 1
            [~,~,~,vperf] = neural_network(data,params);
        case 2
            [~,~,~,vperf] = logistic_regression(data,params);
        case 3
            [~,~,~,vperf] = tree(data,params);
        case 4  
            [~,~,~,vperf] = svm_mdl(data,params);
    end
    
    if vperf<best_vperf
        best_vperf=vperf;
        optimal_vals=[p1 p2 p3];
    end
    if rem(i,5)==0
        fprintf('iter=%d\n',i);
    end
end

