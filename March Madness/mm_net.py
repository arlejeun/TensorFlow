import pandas as pd
import numpy as np
from math import sqrt
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import scipy

############
#PARAMETERS#
############

hidden_units = 5000
hidden_units2 = 1000
hidden_keep = .1
input_keep = 1
lr = .001
beta1 = .9
beta2 = .95
steps = 300

#######
#SETUP#
#######

#Load data
madness_data = 'madness_avg.csv'
madness = pd.read_csv(madness_data)
#print madness.corr()

#Train on 2013 and 2014 to predict 2015
prediction_year = 2015
prior_year_mask = madness['Year'] != prediction_year
trX, teX = madness[prior_year_mask], madness[~prior_year_mask]
trY, teY = trX.pop('Performance'), teX.pop('Performance')
trY, teY = np.reshape(trY, (len(trY),1)), np.reshape(teY, (len(teY),1))

#Scale
train_teams = trX.pop('Name')
test_teams = teX.pop('Name')
test_seeds = teX['Seed']
trX = preprocessing.scale(trX)
teX = preprocessing.scale(teX)

feature_cols = 30
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

test_error = []
train_error = []
prec = []
rec = []
f1 = []

for i in range(steps):

    if i % 25 == 0:

        test_feed = {X: teX, Y: teY, p_keep_input: 1.0, p_keep_hidden:1.0}
        train_feed = {X: trX, Y: trY, p_keep_input: 1.0, p_keep_hidden:1.0}
        print sqrt(2*sess.run(cost,feed_dict=train_feed)/len(trX)), sqrt(2*sess.run(cost,feed_dict=test_feed)/len(teX))

    sess.run(train_op, feed_dict={X: trX, Y: trY, p_keep_input: input_keep, p_keep_hidden: hidden_keep})

############
#PREDICTION#
############

predictions = sess.run(py_x,feed_dict={X:teX,p_keep_input: 1.0, p_keep_hidden: 1.0})
zipped = map(lambda x :(x[0][0],x[1][0],x[2],x[3]),zip(teY,predictions,test_teams,test_seeds))
sorted_by_prediction = sorted(zipped, key=lambda x:x[1])
sorted_by_reality = sorted(zipped, key=lambda x:x[0])
ranked_zip = map(lambda x: (sorted_by_prediction.index(x),sorted_by_reality.index(x)) ,zipped)
print "(Real, Model, Team, Seed)", sorted_by_reality[::-1]
print "(Real, Model, Team, Seed)", sorted_by_prediction[::-1]
print "Cor, p", scipy.stats.pearsonr(*zip(*ranked_zip))
plt.scatter(*zip(*ranked_zip))
plt.show()

sess.close()
