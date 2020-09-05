function visualize_data(data_struct, varargin)
%choosing 3 variables, A, B and C
%if not defined, will be chosen defaultly
if ~isempty(varargin)
    [colX,colY,colZ] = varargin{:};
else
    colX = 4 ; colY=10 ; colZ=12; %day of the week, apparent temerture, wind
end
 
samples = data_struct.train_x;
labels = data_struct.train_y;

pos_ind = find(labels == 1);
neg_ind = find(labels == 0);

%pos samples
X = samples(pos_ind,colX);
Y = samples(pos_ind,colY);
Z = samples(pos_ind,colZ);

%neg samples
x = samples(neg_ind,colX);
y = samples(neg_ind,colY);
z = samples(neg_ind,colZ);


figure
scatter3(X,Y.*50,Z.*60,'MarkerEdgeColor','k','MarkerFaceColor','g');
hold on
scatter3(x,y.*50,z.*60,'MarkerEdgeColor','k','MarkerFaceColor','r')
set(gca,'XLim',[1 12]) 
%set(get(gca,'YLabel'),'Position',[0.5325   -1.6496   17.5000]);
title('Bike Data in Different Days','FontSize',15)%%%%%%%%to change
legend('Popular Riding Day','Unpopular Riding Day')
xlabel('Month')
ylabel('Apparent Tempartue (celsius)')
zlabel('Wind Speed (km/h)')
view(32.9000,16.5600);


