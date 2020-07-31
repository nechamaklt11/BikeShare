function visualize_data(data_struct, varargin)
%choosing 3 variables, A, B and C
%if not defined, will be chosen defaultly
if ~isempty(varargin)
    [colX,colY,colZ] = varargin{:};
else
%     colX = 6 ; colY=12 ; colZ=14;
    colX = 5 ; colY=11 ; colZ=13;
end
 
samples = data_struct.train_x;
labels = data_struct.train_y;
num_samples = length(labels);

pos_ind = find(labels == 1);
neg_ind = find(labels == 0);

%pos samples
X = [samples{pos_ind,colX}];
Y = [samples{pos_ind,colY}];
Z = [samples{pos_ind,colZ}];

%neg samples
x = [samples{neg_ind,colX}];
y = [samples{neg_ind,colY}];
z = [samples{neg_ind,colZ}];


figure
h=scatter3(X,Y,Z,'MarkerEdgeColor','k','MarkerFaceColor','g')
hold on
scatter3(x,y,z,'MarkerEdgeColor','k','MarkerFaceColor','r')

title('This is a title','FontSize',15)%%%%%%%%to change
legend('Popular Riding Time','Unpopular Riding Time')
xlabel('propertyA')
ylabel('propertyB')
zlabel('propertyC')

