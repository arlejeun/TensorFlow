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
hidden_units2 = 2000
hidden_keep = .1
input_keep = .8
lr = .001
beta1 = .9
beta2 = .95
steps = 100

#######
#SETUP#
#######

#Load data
madness_data = 'madness_avg.csv'
madness = pd.read_csv(madness_data)
#print madness.corr()

#Train on every year but one, and predict that year
prediction_year = 2012
prior_year_mask = madness['Year'] != prediction_year
trX, teX = madness[prior_year_mask], madness[~prior_year_mask]

#Split to work with multiple probabilities. Assume 8 values mapped to 4.
#Bucket teams into early losses, middle losses, and late losses
#Splitting into all eight wasn't working, likely because we'd overfit champion predictions
#Hopefully clumping final four contenders together will address this
def split_y(y):
    split = [0]*4
    if y==1 or y==2: split[0] = 1       #One and done - 36 Teams per year
    elif y==3: split[1] = 1             #Thirty-two - 16 Teams per year
    elif y==4: split[2] = 1             #Sweet Sixteen - 8 Teams per year
    else: split[3] = 1                  #Elite Eight - 8 Teams per year
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

feature_cols = 30
output_vals = 4

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
train_op = tf.train.AdamOptimizer(lr, beta1, beta2).minimize(cost)

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
        print sess.run(cost,feed_dict=train_feed), sess.run(cost,feed_dict=test_feed)

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

predictions = sess.run(predict_op,feed_dict={X:teX,p_keep_input: 1.0, p_keep_hidden: 1.0})

#A zipped tuple has (Snake, Team, Model, Real)
average_round = [2,3,4,6]
zipped = map(lambda x :(x[0],x[1],sum([average_round[i]*y for i,y in enumerate(x[2])]),x[3]),zip(test_snake,test_teams,predictions,test_perf))

sorted_by_prediction = sorted(zipped, key=lambda x:x[2])
sorted_by_snake = sorted(zipped, key=lambda x:x[0],reverse=True)

sorted_by_prediction = assign_points(sorted_by_prediction)
sorted_by_snake = assign_points(sorted_by_snake)

print "(Snake, Team, Model, Real, SnakePred)", sorted_by_snake[::-1]
print "(Snake, Team, Model, Real, ModelPred)", sorted_by_prediction[::-1]

print "Snake baseline score: ", score_bracket(sorted_by_snake)
print "Neural net score: ", score_bracket(sorted_by_prediction)

output = sorted(map(lambda x :(x[0],x[1],list(x[2]),x[3]),zip(test_snake,test_teams,predictions,test_perf)), key=lambda x:x[2][-1], reverse=True)
print "Raw Output"
for o in output:
    print o

sess.close()