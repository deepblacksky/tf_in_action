import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from DenoisingAutoencoder import AdditiveGaussianNoiseAutoEncoder

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def stander_scale(X_train, Y_train):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    Y_train = preprocessor.transform(Y_train)
    return X_train, Y_train


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

# 标准化
X_train, X_test = stander_scale(mnist.train.images, mnist.train.images)

# 定义几个参数
n_samples = int(mnist.train.num_examples)
# 训练轮数
training_epochs = 40
batch_size = 128
display_step = 1
hidden_node_num = 500

autoencoder = AdditiveGaussianNoiseAutoEncoder(n_input=784,
                                               n_hidden=hidden_node_num,
                                               transfer_function=tf.nn.softplus,
                                               optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                               scale=0.01)
cost_list = []
# training
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size

    cost_list.append(avg_cost)

    # if epoch % display_step == 0:
    #     print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

fig = plt.figure()
fig.add_subplot(1, 1, 1)
plt.tight_layout()
# plt.scatter(np.arange(1, training_epochs+1), cost_list, c='b', marker=".")
plt.plot(cost_list)
plt.title("training_epochs = %d, batch_size = %d, hidden_node_num = %d" %
          (training_epochs, batch_size, hidden_node_num))
# plt.show()
plt.savefig("avg_cost.png")

print("Total Cost:" + str(autoencoder.calc_total_cost(X_test)))
