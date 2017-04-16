import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
sess = tf.InteractiveSession()

# input node
in_units = 784
# hidden layer node
h1_units = 300
# hidden layer weights
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
# hidden layer bias
b1 = tf.Variable(tf.zeros([h1_units]))
# output weight
W2 = tf.Variable(tf.zeros([h1_units, 10]))
# output bias
b2 = tf.Variable(tf.zeros([10]))

# input data
x = tf.placeholder(tf.float32, [None, in_units])
# dropout rate
keep_prob = tf.placeholder(tf.float32)

# Training NN by TensorFlow
# **************************STEP 1**************************
# define algorithm formula

hidden1 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.add(tf.matmul(hidden1, W2), b2))

# **************************STEP 2**************************
# define loss function and choose optimizer

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entry = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entry)

# **************************STEP 3**************************
# training (dropout)

tf.global_variables_initializer().run()
accuracy_list = []
display_step = 100
for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})

# **************************STEP 4**************************
# accuracy evaluation (prediction)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc = accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    accuracy_list.append(acc)
    if i % display_step == 0:
        print(acc)

fig = plt.figure()
fig.add_subplot(1, 1, 1)
plt.title("hidden_node=%d, batch_size=100, epoch=%d" % (h1_units, int(3000*100/300000)))
plt.plot(accuracy_list)
plt.savefig("accuracy.png")
plt.show()

