import input_data
import tensorflow as tf

##############
#SETUP
##############

#data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder("float", [None, 784], name="x-input")
W = tf.Variable(tf.zeros([784,10]), name="weights")
b = tf.Variable(tf.zeros([10], name="bias"))

# use a name scope to organize nodes in the graph visualizer
with tf.name_scope("Wx_b") as scope:
  y = tf.nn.softmax(tf.matmul(x,W) + b)

# Add summary ops to collect data
w_hist = tf.histogram_summary("weights", W)
b_hist = tf.histogram_summary("biases", b)
y_hist = tf.histogram_summary("y", y)

# Define loss and optimizer
y_ = tf.placeholder("float", [None,10], name="y-input")
# More name scopes will clean up the graph representation
with tf.name_scope("xent") as scope:
  cross_entropy = -tf.reduce_sum(y_*tf.log(y))
  ce_summ = tf.scalar_summary("cross entropy", cross_entropy)
with tf.name_scope("train") as scope:
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

with tf.name_scope("test_set") as scope:
  correct_prediction_test = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accuracy_test = tf.reduce_mean(tf.cast(correct_prediction_test, "float"))
  accuracy_test_summary = tf.scalar_summary("test_accuracy", accuracy_test)

with tf.name_scope("train_set") as scope:
  correct_prediction_train = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accuracy_train = tf.reduce_mean(tf.cast(correct_prediction_train, "float")) -.1
  accuracy_summary = tf.scalar_summary("train_accuracy", accuracy_train)

# Merge all the summaries and write them out to /tmp/mnist_logs
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/mnist_logs", sess.graph_def)
tf.initialize_all_variables().run()

# Train the model, and feed in test data and record summaries every 10 steps

for i in range(100):
  if i % 10 == 0:  # Record summary data, and the accuracy
    test_feed = {x: mnist.test.images, y_: mnist.test.labels}
    test_result = sess.run([merged, accuracy_test], feed_dict=test_feed)
    test_summary_str = test_result[0]
    test_acc = test_result[1]
    writer.add_summary(test_summary_str, i)

    train_feed = {x: mnist.train.images, y_: mnist.train.labels}
    train_result = sess.run([merged, accuracy_train], feed_dict=train_feed)
    train_summary_str = train_result[0]
    train_acc = train_result[1]
    writer.add_summary(train_summary_str, i)
    print("Accuracy at step %s: %s, %s" % (i, train_acc, test_acc))
  else:
    batch_xs, batch_ys = mnist.train.next_batch(100)
    feed = {x: batch_xs, y_: batch_ys}
    sess.run(train_step, feed_dict=feed)


