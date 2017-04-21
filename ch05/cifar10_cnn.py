import cifar10_input
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import math
import os
import sys
import tarfile
from six.moves import urllib

max_steps = 20000
batch_size = 128
data_dir = "/home/yuxin/PycharmProjects/tf/cifar10_data/cifar-10-batches-bin"
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def maybe_download_and_extract():
    """
    Download and extract the tarball from Alex's website.
    """

    dest_directory = "/home/yuxin/PycharmProjects/tf/cifar10_data"
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


# init weights, add L2 regularization
def variable_with_weight_loss(shape, stddev, wl):
    """
    
    :param shape: 
    :param stddev: 
    :param wl: 
    :return: 
    """
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name="weight_loss")
        tf.add_to_collection('losses', weight_loss)
    return var


# download dataset
maybe_download_and_extract()
image_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)

# create test data
image_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

# input placeholder
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

# define conv1 layer
weight1 = variable_with_weight_loss([5, 5, 3, 64], stddev=0.05, wl=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
# LRN
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# define covn2
# color channel:3->64, bias:0->0.1,LRN->MAX_POOL
weight2 = variable_with_weight_loss([5, 5, 64, 64], stddev=0.05, wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, strides=[1, 1, 1, 1], padding="SAME")
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

# 3rd layer:fcl,384 node
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss([dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

# 4th layer:fcl:192 node
weight4 = variable_with_weight_loss([384, 192], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

# last layer:(no softmax),output result
weight5 = variable_with_weight_loss([192, 10], stddev=1.0/192, wl=0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)


# calculate loss, cross_entropy
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name="croos_entropy_per_example")
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")
    tf.add_to_collection("losses", cross_entropy_mean)

    return tf.add_n(tf.get_collection("losses"), name="total_loss")


loss = loss(logits, label_holder)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Data Augmentation thread
tf.train.start_queue_runners()

# training
loss_list = []
for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([image_train, labels_train])
    _, loss_value = sess.run([train_op, loss], feed_dict={image_holder: image_batch, label_holder: label_batch})
    duration = time.time() - start_time
    loss_list.append(loss_value)
    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)
        format_str = "step %d,loss=%.2f (%.1f examples/sec; %.3f sec/batch)"
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

fig = plt.figure()
fig.add_subplot(1, 1, 1)
plt.tight_layout()
plt.plot(loss_list)
# plt.show()
plt.savefig("loss_cnn_cifar10.png")

# test prediction accuracy
num_examples = 10000
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([image_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch, label_holder: label_batch})
    true_count += np.sum(predictions)
    step += 1

precision = true_count / total_sample_count
print("precision @ 1 = %.3f" % precision)

