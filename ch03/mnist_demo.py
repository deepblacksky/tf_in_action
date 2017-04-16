from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def my_mnist():
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    # implement Softmax Regression
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, ([None, 10]))
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    # init
    tf.global_variables_initializer().run()
    acc = []
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        train_step.run({x: batch_xs, y_: batch_ys})
        # accuracy
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
        acc.append(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(acc)
    plt.show()

if __name__ == '__main__':
    # print(mnist.train.images.shape, mnist.train.labels.shape)
    # print(mnist.test.images.shape, mnist.test.labels.shape)
    # print(mnist.validation.images.shape, mnist.validation.labels.shape)
    my_mnist()


