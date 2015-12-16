import itertools
from multiprocessing import Pool
import json
import amr_net

############
#PARAMETERS#
############

NUM_PROCESS = 4

hidden_units_grid = [16, 64, 256, 1024, 4096]
input_keep_grid = [.01, .1, .25, .5, .75] 
hidden_keep_grid = [.01, .1, .25, .5, .75]
lr_grid = [.0001, .001, .01, .1, 1]
beta_grid = [.1, .3, .5, .7, .9]
feature_grid = [100, 1000, 10000, 100000, 524684]

hyperparameter_grid = [hidden_units_grid, input_keep_grid, hidden_keep_grid, lr_grid, beta_grid, feature_grid]
hyperparameter_list = list(itertools.product(*hyperparameter_grid))

def main_star(hyperparams):	#pool.map does not support * operator
	return amr_net.main(*hyperparams)

try:
    pool = Pool(NUM_PROCESS)
    output = pool.map(main_star, hyperparameter_list)
finally: # To make sure processes are closed in the end, even if errors happen
    pool.close()
    pool.join()

output_file = 'amr.json'
#Sort on lowest final test error
sorted_output = sorted(output, key=lambda k: k['test_error'][-1]) 
with open(output_file,'w') as f:
	json.dump(sorted_output,f)