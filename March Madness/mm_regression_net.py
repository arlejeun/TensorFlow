import pandas as pd
import numpy as np
from math import sqrt
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

############
#PARAMETERS#
############

hidden_units = 1000
hidden_units2 = 200
hidden_keep = .1
input_keep = 1
lr = .001
rho = .7
steps = 250

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

#Train on every year but one, and predict that year
prediction_year = 2015
prior_year_mask = madness['Year'] != prediction_year
trX, teX = madness[prior_year_mask], madness[~prior_year_mask]
trY, teY = trX.pop('Performance'), teX.pop('Performance')
trY, teY = np.reshape(trY, (len(trY),1)), np.reshape(teY, (len(teY),1))

#Scale
train_teams = trX.pop('Name')
test_teams = teX.pop('Name')
test_seeds = teX['Seed']
test_snake = teX['Snake']
trX = preprocessing.scale(trX)
teX = preprocessing.scale(teX)

feature_cols = 36
output_vals = 1

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

cost = tf.nn.l2_loss(tf.sub(Y,py_x))
#train_op = tf.train.AdamOptimizer(lr, beta1, beta2).minimize(cost)
#train_op = tf.train.FtrlOptimizer(lr).minimize(cost)
train_op = tf.train.RMSPropOptimizer(lr,rho).minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session(
    config=tf.ConfigProto(
        inter_op_parallelism_threads=8,
        intra_op_parallelism_threads=8
    )
)
sess.run(init)

##########
#TRAINING#
##########

for i in range(steps):

    if i % 25 == 0:

        test_feed = {X: teX, Y: teY, p_keep_input: 1.0, p_keep_hidden:1.0}
        train_feed = {X: trX, Y: trY, p_keep_input: 1.0, p_keep_hidden:1.0}
        print sqrt(2*sess.run(cost,feed_dict=train_feed)/len(trX)), sqrt(2*sess.run(cost,feed_dict=test_feed)/len(teX))

    sess.run(train_op, feed_dict={X: trX, Y: trY, p_keep_input: input_keep, p_keep_hidden: hidden_keep})

############
#PREDICTION#
############

#Assumed to take 68 teams
def assign_points(zipped):

    def index_to_perf(i):
        rules = [1]*4 + [2]*32 + [3]*16 + [4]*8 + [5]*4 + [6]*2 + [7] + [8]
        return rules[i]

    return [(x[0], x[1], x[2], x[3], index_to_perf(i)) for i,x in enumerate(zipped)]

def score_bracket(zipped):

    def team_points(real,model):
        #Make play-ins the same as first round
        if real==1: real=2
        if model==1: model=2
        smaller = min(real,model)
        return (2**(smaller-2)) - 1

    return sum([team_points(x[3],x[4]) for x in zipped])

predictions = sess.run(py_x,feed_dict={X:teX,p_keep_input: 1.0, p_keep_hidden: 1.0})

#A zipped tuple has (Seed, Team, Model, Real)
zipped = map(lambda x :(x[0],x[1],x[2][0],x[3][0]),zip(test_snake,test_teams,predictions,teY))

sorted_by_prediction = sorted(zipped, key=lambda x:x[2])
sorted_by_snake = sorted(zipped, key=lambda x:x[0],reverse=True)

sorted_by_prediction = assign_points(sorted_by_prediction)
sorted_by_snake = assign_points(sorted_by_snake)

print "(Snake, Team, Model, Real, SnakePred)", sorted_by_snake[::-1]
print "(Snake, Team, Model, Real, ModelPred)", sorted_by_prediction[::-1]

print "Snake baseline score: ", score_bracket(sorted_by_snake)
print "Neural net score: ", score_bracket(sorted_by_prediction)

output = sorted(map(lambda x :(x[0],x[1],x[2][0],x[3][0]),zip(test_snake,test_teams,predictions,teY)), key=lambda x:x[2], reverse=True)
print "Raw Output"
for o in output:
    print o

sess.close()
