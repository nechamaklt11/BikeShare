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
[num_data,~,all_data] = xlsread(data_file);
rent_num = num_data(:,16);
lim = median(rent_num);  
labels = double(rent_num>lim);

dates =all_data(2:end,2);
for i=1:length(dates)
    dates{i}=str2double(dates{i}(1:2));
end
dates=cell2mat(dates);
edited_data = [dates,num_data(:,3:13)];
% splitting the data to test and training groups
rng(1)
samples_num = size(num_data, 1);
rand_samp_order=randperm(samples_num); %shuffling data set

train_ind = rand_samp_order(1:(round(samples_num*train_per)));
test_ind = rand_samp_order((length(train_ind)+1):samples_num);

%building a structure with all bike info, divided to train and test 
data_struct = struct;
data_struct.train_x = edited_data(train_ind,:);
data_struct.test_x = edited_data(test_ind,:);
data_struct.train_y = labels(train_ind);
data_struct.test_y = labels(test_ind);

data4analysis(data_struct);

function data4analysis(data_struct)
inputs = (data_struct.train_x)';
targets = (data_struct.train_y)';
num_samples = length(targets);
shuffle_ind=randperm(num_samples); %%%%% for i=1:k
lim = round(0.8*num_samples);
test_inputs = (data_struct.test_x)';
test_targets = (data_struct.test_y)';
save('data4analysis','inputs','targets','test_inputs','test_targets')