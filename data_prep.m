function data_struct = data_prep(file_name, train_per)

%labeling data - a day with number of renters who's larger then the median is a
%'popular riding time', the rest are categorized as an "unpopular riding time".
%train per - double between 0 to 1, representing the percentage of data in train group. default=80%
%train group.

switch nargin
    case 1
        train_per=0.8;
end

data_file = file_name;
%[num_data,~,all_data] = xlsread(data_file,1,'A1:Q8646');for hour 
[num_data,~,all_data] = xlsread(data_file);
rent_num = num_data(:,16); %16 (or 17 for hour) is the column of total renters in a day
lim = median(rent_num);  
labels = double(rent_num>lim);
save('all_labels','labels')

% splitting the data to test and training groups
rng(1)
samples_num = size(num_data, 1);
rand_samp_order=randperm(samples_num); %shuffling data set
rand_samp_order=rand_samp_order+1; %fitting the indeces to the all_data cell

train_ind = rand_samp_order(1:(round(samples_num*train_per)));
test_ind = rand_samp_order((length(train_ind)+1):samples_num);

%building a structure with all bike info, divided to train and test 
data_struct = struct;
data_struct.train_x = all_data(train_ind,:);
data_struct.test_x = all_data(test_ind,:);
data_struct.train_y = labels(train_ind-1);
data_struct.test_y = labels(test_ind-1);





