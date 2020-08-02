function accuracy = test_regression(x,t,w)
y = w'*x;
y=logistic_func(y);
y(y<0.5)=0; y(y>=0.5)=1;
diff=y-t;
num_hits= sum(diff==0);
accuracy = (num_hits/length(t))*100;

function g = logistic_func(x)
% logistic function
g=1./(1+exp(-1.*x));

