import pandas as pd
import numpy as np
import itertools
from math import sqrt
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from multiprocessing import Pool

############
#PARAMETERS#
############

NUM_PROCESS = 4
feature_cols = 36
output_vals = 3

#Vary architecture to achieve collection of high variance nets
#Will average predictions of all models
#To be Bayesian, we should be weighting the average based on prior parameter distribution
hidden_units_grid = [250, 500, 1000] * 2
hidden_units2_grid = [250, 500, 1000] * 2
hidden_keep = .1
input_keep = 1
lr = .001
rho = .7
steps = 250

hyperparameter_grid = [hidden_units_grid, hidden_units2_grid]
hyperparameter_list = list(itertools.product(*hyperparameter_grid))

num_nets = len(hyperparameter_list)

#######
#SETUP#
#######

#Load data
madness_data = 'madness_avg.csv'
madness = pd.read_csv(madness_data)

# #Correlations
# cor = madness.corr().abs()
# print cor
# s = cor.unstack()['Performance']
# so = s.sort_values(ascending=False)
# print so

overall_checksum = [0]*output_vals
overall_calibration = {}
years = [2012, 2013, 2014, 2015]
for prediction_year in years:

    print "Starting year", prediction_year
    #Train on every year but one, and predict that year
    prior_year_mask = madness['Year'] != prediction_year
    trX, teX = madness[prior_year_mask], madness[~prior_year_mask]

    #Split to work with multiple probabilities. Assume 8 values mapped to 3.
    #Bucket teams into early losses, middle losses, and late losses
    #Splitting into all eight wasn't working, likely because we'd overfit champion predictions
    #Hopefully clumping elite eight contenders together will address this
    #Also, net wasn't reallt properly differentiating thirty two from sweet sixteen, so 3 makes more sense
    def split_y(y):
        split = [0]*output_vals
        if y==1 or y==2: split[0] = 1       #One and done - 36 Teams per year
        elif y==3 or y==4: split[1] = 1     #Thirty-two, sweet sixteen - 24 Teams per year
        else: split[2] = 1                  #Elite Eight - 8 Teams per year
        return split

    train_perf, test_perf = trX.pop('Performance'), teX.pop('Performance')
    trY = [split_y(y) for y in train_perf]
    teY = [split_y(y) for y in test_perf]

    #Scale
    train_teams = trX.pop('Name')
    test_teams = teX.pop('Name')
    test_seeds = teX['Seed']
    test_snake = teX['Snake']
    trX = preprocessing.scale(trX)
    teX = preprocessing.scale(teX)

    def run_net(trX, teX, trY, teY, hidden_units, hidden_units2):

        net_id = str(hidden_units) + "," + str(hidden_units2)

        ##############
        #ARCHITECTURE#
        ##############

        def init_weights(shape):
            return tf.Variable(tf.random_normal(shape) * 1./sqrt(shape[0]))

        def model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden):
            X = tf.nn.dropout(X, p_drop_input)
            h = tf.nn.relu(tf.matmul(X, w_h))

            h = tf.nn.dropout(h, p_drop_hidden)
            h2 = tf.nn.relu(tf.matmul(h, w_h2))

            h2 = tf.nn.dropout(h2, p_drop_hidden)

            return tf.matmul(h2, w_o)

        X = tf.placeholder("float", [None, feature_cols])
        Y = tf.placeholder("float", [None, output_vals])

        p_keep_input = tf.placeholder("float")
        p_keep_hidden = tf.placeholder("float")

        w_h =  init_weights([feature_cols, hidden_units])
        w_h2 = init_weights([hidden_units, hidden_units2])
        w_o =  init_weights([hidden_units2, output_vals])

        py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x,Y))
        predict_op = tf.nn.softmax(py_x)
        train_op = tf.train.RMSPropOptimizer(lr,rho).minimize(cost)

        init = tf.initialize_all_variables()
        sess = tf.Session(
            config=tf.ConfigProto(
                inter_op_parallelism_threads=4,
                intra_op_parallelism_threads=4
            )
        )
        sess.run(init)

        ##########
        #TRAINING#
        ##########

        for i in range(steps):

            # if i % 120 == 0:

            #     test_feed = {X: teX, Y: teY, p_keep_input: 1.0, p_keep_hidden:1.0}
            #     train_feed = {X: trX, Y: trY, p_keep_input: 1.0, p_keep_hidden:1.0}
            #     print net_id, i, sess.run(cost,feed_dict=train_feed), sess.run(cost,feed_dict=test_feed)

            sess.run(train_op, feed_dict={X: trX, Y: trY, p_keep_input: input_keep, p_keep_hidden: hidden_keep})

        ############
        #PREDICTION#
        ############

        predictions = sess.run(predict_op,feed_dict={X:teX,p_keep_input: 1.0, p_keep_hidden: 1.0})
        sess.close()

        #########
        #SUMMARY#
        #########

        final_rankings = {}

        temp_pred = zip(test_seeds,test_teams,predictions,test_perf)
        for team in temp_pred:
            team_id = "(" + str(team[0]) + ")" + " " + team[1] + ": " + str(team[3])
            if team_id in final_rankings:
                final_rankings[team_id] = [sum(x) for x in zip(list(team[2]),final_rankings[team_id])]
            else:
                final_rankings[team_id] = list(team[2])

        print "Finishing net", net_id
        return final_rankings


    #############
    #PARALLELIZE#
    #############

    def main_star(hyperparams): #pool.map does not support * operator
        return run_net(trX, teX, trY, teY, *hyperparams)

    try:
        pool = Pool(NUM_PROCESS)
        output = pool.map(main_star, hyperparameter_list)
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    #########
    #RESULTS#
    #########

    #Average results and generate composite score
    average_rankings = {}
    for o in output:
        for team in o:
            if team in average_rankings:
                average_rankings[team] = [average_rankings[team][i] + o[team][i] for i in range(output_vals)]
            else:
                average_rankings[team] = o[team]

    average_round = [2,3.333,5.875]
    for team in average_rankings:
        average_rankings[team] = [round(prob / num_nets,4) for prob in average_rankings[team]]
        average_rankings[team] = (round(sum([average_round[i]*y for i,y in enumerate(average_rankings[team])]),4), average_rankings[team])

    #Sort by composite score and print
    checksum = [0]*output_vals
    calibration = {}
    sorted_by_results = sorted(average_rankings.items(),key=lambda x: x[1][0],reverse=True)
    for team in sorted_by_results:
        print team[0], team[1]

        #checksum represents marginal probabilities of buckets over all teams
        #checksum should reflect actual number of teams allowed per bucket
        checksum = [checksum[i]+team[1][1][i] for i in range(output_vals)]

        #calibration represents how often a prediction is correct based on certainty of prediction
        actual = int(team[0][-1])
        actual_index = np.argmax(split_y(actual))
        for i,prob in enumerate(team[1][1]):
            tenth = round(prob,1)
            if tenth in calibration:
                calibration[tenth] = (calibration[tenth][0], calibration[tenth][1]+1)
            else:
                calibration[tenth] = (0,1)
            if i == actual_index:
                calibration[tenth] = (calibration[tenth][0]+1,calibration[tenth][1])
        
    for tenth in calibration:
        calibration[tenth] = (calibration[tenth][0] / float(calibration[tenth][1]), calibration[tenth])
        if tenth in overall_calibration:
            success = overall_calibration[tenth][1][0]+calibration[tenth][1][0]
            trial = overall_calibration[tenth][1][1]+calibration[tenth][1][1]
            overall_calibration[tenth] = (float(success)/trial, (success,trial))
        else:
            overall_calibration[tenth] = calibration[tenth]

    overall_checksum = [v + checksum[i] for i,v in enumerate(overall_checksum)]

#Check internal and external calibration
#Expect checksum to reflect actual number of teams per bucket
#Expect probs to reflect actual results
print "Marginal sums across teams: ", [x/len(years) for x in overall_checksum]
print "Prediction certainty vs. accuracy: "
sorted_by_certainty = sorted(overall_calibration.items(),key=lambda x: float(x[0]))
for x in sorted_by_certainty:
    print x
