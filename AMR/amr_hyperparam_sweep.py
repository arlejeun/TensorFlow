import pandas as pd
import numpy as np
import itertools
from multiprocessing import Pool
import json
import random
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import amr_net

############
#PARAMETERS#
############

NUM_PROCESS = 4

hidden_units_grid = [250,500,750,1000]
hidden_units2_grid = [250,500,750,1000]
input_keep_grid = [.1] # [.01, .1, .25, .5, .75] 
hidden_keep_grid = [.1] # [.01, .1, .25, .5, .75]
lr_grid = [.005] # [.0001, .001, .01, .1, 1]
beta_grid = [.9] #[.1, .3, .5, .7, .9]

features = 10000

hyperparameter_grid = [hidden_units_grid, hidden_units2_grid, input_keep_grid, hidden_keep_grid, lr_grid, beta_grid]
hyperparameter_list = list(itertools.product(*hyperparameter_grid))

#######
#SETUP#
#######


tab_data = 'carbapenem.k10'

#Get targets and encode as integers
response_row = pd.read_csv(tab_data, sep='\t', nrows=1, header=None)
response_col = np.array(response_row.transpose()[1:])

#Format to work with multi-class classifier
def split_y(y):
    if y == 1: return [0,1]
    else: return [1,0] 

amr_data_y = np.array(map(split_y,response_col))
test_rows = random.sample(range(len(response_col)), len(response_col) / 3)
test_row_mask = np.array([x in test_rows for x in range(len(response_col))])
train_row_mask = np.array([not x for x in test_row_mask])

def main_star(hyperparams):	#pool.map does not support * operator, also we need to get different feature set for each net
	
	print "loading"

	#Too many columns for this machine, randomly select a subset
	col_limit = features
	col_total = 524684
	skip_cols = [0]+random.sample(range(1,col_total), col_total-col_limit) #Always skip row 0
	df = pd.read_csv(tab_data, sep='\t', skiprows=skip_cols, header=None)

	# Transpose, relabel, and scale
	amr_data_x = df.transpose()[1:]
	feature_cols = len(amr_data_x.columns)
	amr_data_x = preprocessing.scale(amr_data_x)

	trX, teX, trY, teY = amr_data_x[train_row_mask], amr_data_x[test_row_mask], amr_data_y[train_row_mask], amr_data_y[test_row_mask]

	return amr_net.main(trX, teX, trY, teY, feature_cols, *hyperparams)

try:
    pool = Pool(NUM_PROCESS)
    output = pool.map(main_star, hyperparameter_list)
finally: # To make sure processes are closed in the end, even if errors happen
    pool.close()
    pool.join()


#Performance
response = np.argmax(amr_data_y[test_row_mask], axis=1)

for i,pred in enumerate(output):
	prediction = np.argmax(pred,axis=1)
	print prediction
	print "Run " + str(i+1) + ": " + str(1 - sum(response == prediction) / float(len(response)))

#Average
avg_output = np.average(output, axis=0)
avg_prediction = np.argmax(avg_output,axis=1)

print avg_prediction
print avg_output

print "Average: " + str(1 - sum(response == avg_prediction) / float(len(response)))

# output_file = 'amr.json'
# #Sort on lowest final test error
# sorted_output = sorted(output, key=lambda k: k['test_error'][-1]) 
# with open(output_file,'w') as f:
# 	json.dump(sorted_output,f)