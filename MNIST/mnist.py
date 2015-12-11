import input_data
import tensorflow as tf

##############
#SETUP
##############

#data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#placeholders
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

#parameters
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#model
y = tf.nn.softmax(tf.matmul(x, W) + b)

#cost
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#train
lr = .01
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

##############
#LAUNCH
##############

#Setup
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#Training
steps = 1000
batch = 100
for i in range(steps):
  batch_xs, batch_ys = mnist.train.next_batch(batch)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#Evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})


