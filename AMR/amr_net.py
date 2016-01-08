import tensorflow as tf
from math import sqrt

############
#PARAMETERS#
############

steps = 51

def main(trX, teX, trY, teY, feature_cols, hidden_units, hidden_units2, input_keep, hidden_keep, lr, beta1):

    ##############
    #ARCHITECTURE#
    ##############

    print "building"

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
    Y = tf.placeholder("float", [None, 2])

    p_keep_input = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")

    w_h =  init_weights([feature_cols, hidden_units])
    w_h2 =  init_weights([hidden_units, hidden_units2])
    w_o =  init_weights([hidden_units2, 2])

    py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    train_op = tf.train.AdamOptimizer(lr, beta1).minimize(cost)
    predict_op = tf.nn.softmax(py_x)

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

    print "training"

    test_error = []
    train_error = []
    prec = []
    rec = []
    f1 = []

    for i in range(steps):

        if i % 5 == 0:  # Record test/train error and other scores
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

            print hidden_units, hidden_units2, i, train_result, test_result
        
        #Too few observations for full batch to be a problem
        sess.run(train_op, feed_dict={X: trX, Y: trY, p_keep_input: input_keep, p_keep_hidden: hidden_keep})

    predictions = sess.run(predict_op, feed_dict=test_feed)
    sess.close()
    return predictions
