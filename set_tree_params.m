function params = set_tree_params()
params=struct('MaxNumSplits', 25, 'MinLeafSize',1, 'MaxNumCategories', 10, 'Prune', 'on', 'plot', true);
