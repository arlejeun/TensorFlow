import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import random
from math import sqrt
import tensorflow as tf

#######
#SETUP#
#######

tab_data = 'carbapenem.k10'

#Too many columns for this machine, randomly select a subset
col_limit = 10000
col_total = 524684
skip_cols = [0]+random.sample(range(1,col_total), col_total-col_limit) #Always skip row 0
df = pd.read_csv(tab_data, sep='\t', skiprows=skip_cols, header=None)

# Transpose, relabel, and scale
amr_data_x = df.transpose()[1:]
feature_cols = len(amr_data_x.columns)
amr_data_x = preprocessing.scale(amr_data_x)

#Get targets and encode as integers
response_row = pd.read_csv(tab_data, sep='\t', nrows=1, header=None)
response_col = np.array(response_row.transpose()[1:])

#Format to work with multi-class classifier
def split_y(y):
	if y == 1: return [0,1]
	else: return [1,0] 

amr_data_y = np.array(map(split_y,response_col))

############
#PARAMETERS#
############

training_pct = 75
hidden_units = 250
input_keep = .1
hidden_keep = .3
lr = .001
rho = .9
steps = 101

##############
#ARCHITECTURE#
##############

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape) * 1./sqrt(shape[0]))

def model(X, w_h, w_o, p_drop_input, p_drop_hidden):
    X = tf.nn.dropout(X, p_drop_input)
    h = tf.nn.relu(tf.matmul(X, w_h))
    h = tf.nn.dropout(h, p_drop_hidden)

    return tf.matmul(h, w_o)

sess = tf.InteractiveSession()

X = tf.placeholder("float", [None, feature_cols])
Y = tf.placeholder("float", [None, 2])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

w_h =  init_weights([feature_cols, hidden_units])
w_o =  init_weights([hidden_units, 2])

py_x = model(X, w_h, w_o, p_keep_input, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(lr, rho).minimize(cost)
predict_op = tf.argmax(py_x, 1)

#############
#TENSORBOARD#
#############

#Error
with tf.name_scope("error") as scope:
    correct_prediction = tf.equal(tf.argmax(py_x,1), tf.argmax(Y,1))
    error = 1-tf.reduce_mean(tf.cast(correct_prediction, "float"))
    error_summary = tf.scalar_summary("error", error)

#Precision
with tf.name_scope("precision") as scope:
    tp = tf.logical_and(tf.equal(tf.argmax(py_x,1),1),tf.equal(tf.argmax(Y,1),1))
    fp = tf.logical_and(tf.equal(tf.argmax(py_x,1),1),tf.equal(tf.argmax(Y,1),0))
    precision = tf.div(tf.reduce_sum(tf.cast(tp, "float")) ,tf.add(tf.reduce_sum(tf.cast(tp, "float")), tf.reduce_sum(tf.cast(fp, "float"))))
    precision_summary = tf.scalar_summary("precision", precision)

#Recall
with tf.name_scope("recall") as scope:
    tp = tf.logical_and(tf.equal(tf.argmax(py_x,1),1),tf.equal(tf.argmax(Y,1),1))
    fn = tf.logical_and(tf.equal(tf.argmax(py_x,1),0),tf.equal(tf.argmax(Y,1),1))
    recall = tf.div(tf.reduce_sum(tf.cast(tp, "float")), tf.add(tf.reduce_sum(tf.cast(tp, "float")), tf.reduce_sum(tf.cast(fn, "float"))))
    recall_summary = tf.scalar_summary("recall", recall)

#F1 score
with tf.name_scope("f1_score") as scope:
    f1_score = 2 * tf.div(tf.mul(precision,recall),tf.add(precision,recall))
    f1_summary = tf.scalar_summary("f1_score", f1_score)


#Merge all the summaries and write them out to /tmp/amr_logs
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/amr_logs", sess.graph_def)
tf.initialize_all_variables().run()

##########
#TRAINING#
##########

trX, teX, trY, teY = train_test_split(amr_data_x, amr_data_y, train_size=training_pct/100.0)

for i in range(steps):

    if i % 10 == 0:  # Record summary data, and the test/train error
        test_feed = {X: teX, Y: teY, p_keep_input: 1.0, p_keep_hidden:1.0}
        test_result = sess.run([merged, error], feed_dict=test_feed)
        test_summary_str = test_result[0]
        test_err = test_result[1]
        writer.add_summary(test_summary_str, i)

        train_feed = {X: trX, Y: trY, p_keep_input: 1.0, p_keep_hidden:1.0}
        train_result = sess.run([merged, error], feed_dict=train_feed)
        train_summary_str = train_result[0]
        train_err = train_result[1]
        writer.add_summary(train_summary_str, i)

        precision_result = sess.run([merged, precision], feed_dict=test_feed)  
        precision_summary_str = precision_result[0]
        prec = precision_result[1]
        writer.add_summary(precision_summary_str, i)      

        recall_result = sess.run([merged, recall], feed_dict=test_feed)  
        recall_summary_str = recall_result[0]
        rec = recall_result[1]
        writer.add_summary(recall_summary_str, i)      

        f1_result = sess.run([merged, f1_score], feed_dict=test_feed)  
        f1_summary_str = f1_result[0]
        f1 = f1_result[1]
        writer.add_summary(f1_summary_str, i)      
        
        print("Train Error, Test Error, Precision, Recall, F1 at step %s: %s, %s, %s, %s, %s" % (i, train_err, test_err, prec, rec, f1))
    
    #Too few observations for full batch to be a problem
    sess.run(train_op, feed_dict={X: trX, Y: trY, p_keep_input: input_keep, p_keep_hidden: hidden_keep})
