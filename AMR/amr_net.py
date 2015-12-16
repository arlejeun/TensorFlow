import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import random
from math import sqrt
import tensorflow as tf

############
#PARAMETERS#
############

training_pct = 75
steps = 101

def main(hidden_units, input_keep, hidden_keep, lr, beta1, features):

    #######
    #SETUP#
    #######

    tab_data = 'carbapenem.k10'

    #Too many columns for this machine, randomly select a subset
    col_limit = features
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

    X = tf.placeholder("float", [None, feature_cols])
    Y = tf.placeholder("float", [None, 2])

    p_keep_input = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")

    w_h =  init_weights([feature_cols, hidden_units])
    w_o =  init_weights([hidden_units, 2])

    py_x = model(X, w_h, w_o, p_keep_input, p_keep_hidden)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    train_op = tf.train.AdamOptimizer(lr, beta1).minimize(cost)
    predict_op = tf.argmax(py_x, 1)

    #############
    #TENSORBOARD#
    #############

    #Base metrics
    tp = tf.logical_and(tf.equal(tf.argmax(py_x,1),1),tf.equal(tf.argmax(Y,1),1))
    fp = tf.logical_and(tf.equal(tf.argmax(py_x,1),1),tf.equal(tf.argmax(Y,1),0))
    fn = tf.logical_and(tf.equal(tf.argmax(py_x,1),0),tf.equal(tf.argmax(Y,1),1))

    #Error
    correct_prediction = tf.equal(tf.argmax(py_x,1), tf.argmax(Y,1))
    error = 1-tf.reduce_mean(tf.cast(correct_prediction, "float"))
    error_summary = tf.scalar_summary("error", error)

    #Precision
    precision = tf.div(tf.reduce_sum(tf.cast(tp, "float")) ,tf.add(tf.reduce_sum(tf.cast(tp, "float")), tf.reduce_sum(tf.cast(fp, "float"))))
    precision_summary = tf.scalar_summary("precision", precision)

    #Recall
    recall = tf.div(tf.reduce_sum(tf.cast(tp, "float")), tf.add(tf.reduce_sum(tf.cast(tp, "float")), tf.reduce_sum(tf.cast(fn, "float"))))
    recall_summary = tf.scalar_summary("recall", recall)

    #F1 score
    f1_score = 2 * tf.div(tf.mul(precision,recall),tf.add(precision,recall))
    f1_summary = tf.scalar_summary("f1_score", f1_score)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    ##########
    #TRAINING#
    ##########

    trX, teX, trY, teY = train_test_split(amr_data_x, amr_data_y, train_size=training_pct/100.0)
    test_error = []
    train_error = []
    prec = []
    rec = []
    f1 = []

    for i in range(steps):

        if i % 10 == 0:  # Record test/train error and other scores
            test_feed = {X: teX, Y: teY, p_keep_input: 1.0, p_keep_hidden:1.0}
            train_feed = {X: trX, Y: trY, p_keep_input: 1.0, p_keep_hidden:1.0}

            test_result = sess.run(error, feed_dict=test_feed)
            train_result = sess.run(error, feed_dict=train_feed)
            precision_result = sess.run(precision, feed_dict=test_feed)     
            recall_result = sess.run(recall, feed_dict=test_feed)       
            f1_result = sess.run(f1_score, feed_dict=test_feed)  

            #These need to be floats to dump as json later
            test_error.append(float(test_result))
            train_error.append(float(train_result))   
            prec.append(float(precision_result))
            rec.append(float(recall_result))
            f1.append(float(f1_result))
        
        #Too few observations for full batch to be a problem
        sess.run(train_op, feed_dict={X: trX, Y: trY, p_keep_input: input_keep, p_keep_hidden: hidden_keep})

    return {
            'test_error':test_error,
            'train_error':train_error, 
            'Precision':prec, 
            'Recall':rec, 
            'F1':f1, 
            'HU':hidden_units, 
            'IK':input_keep, 
            'HK':hidden_keep, 
            'LR':lr, 
            'Beta':beta1,
            'feature_count':features
            }

    sess.close()
