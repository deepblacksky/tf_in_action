"""

implement a simple CNN, dataSet:MNIST

conv1 --> conv2 --> FCL

"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
sess = tf.InteractiveSession()


# init weights and bias
# we need to add some noise to weight to break completely symmetrical
# so, wo use truncated normal noise:std=0.1
def weights_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# due to using ReLU function, we need add little positive(0.1) to bias. It can avoid dead neurons
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# create conv function and pool function
def conv2d(x, W):
    """
    :param x: inputData
    :param W: Filter:[a,b,c,d], a,b:convolution kernel size, c:color channel(RGB image:3,gray:1), d:conv_kernel number
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# define input
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
# tf.reshape: x:input tensor(1D), [-1, 28, 28, 1]:output shape
# "-1":sample number not sure, "28,28":output size(2D), "1":color channel(RGB:3,grey:1)
x_image = tf.reshape(x, [-1, 28, 28, 1])

# define layer1
# conv_kernel size:5x5, color channel:1, conv_kernel:32
W_conv1 = weights_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# define layer2
# conv_kernel size:5x5, channel:32, conv_kernel:64
W_conv2 = weights_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Full connection layer
# after twice max pool, image size:28/2/2=7, conv2 kernel:64, so output image size:7*7*64
# FCL hidden node:1024
W_fc1 = weights_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# 2D --> 1D
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout avoid overFit
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# softmax layer
W_fc2 = weights_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# define loss function and optimizer(learning rate = 1e-4)
cross_entry = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entry)

# accuracy
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training
tf.global_variables_initializer().run()
accuracy_list = []
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        accuracy_list.append(train_accuracy)
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# plot
fig = plt.figure()
fig.add_subplot(1, 1, 1)
plt.tight_layout()
plt.plot(accuracy_list)
plt.title("CNN conv1:32@5x5,conv2:64@5x5,fc1:1024")
# plt.show()
plt.savefig("accuracy_simple_cnn_mnist.png")

# test training
# print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
# this will out of GPU memory, so we change batch test
accuracy_sum = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
good = 0
total = 0
for i in range(10):
    testSet = mnist.test.next_batch(50)
    good += accuracy_sum.eval(feed_dict={x: testSet[0], y_: testSet[1], keep_prob: 1.0})
    total += testSet[0].shape[0]
print("test accuracy %g" % (good/total))

# another answer
# for i in range(10):
#     testSet = mnist.test.next_batch(50)
#     print("test accuracy %g" % accuracy.eval(feed_dict={ x: testSet[0], y_: testSet[1], keep_prob: 1.0}))
