function params = set_tree_params()
params=struct('MaxNumSplits', 600, 'MinLeafSize',2, 'MaxNumCategories', 8, 'Prune', 'on', 'plot', false);
