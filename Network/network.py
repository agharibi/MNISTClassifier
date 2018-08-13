import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

class Network:
    @staticmethod
    def weight_variable(shape, name=None):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial, name=name)

    @staticmethod
    def bias_variable(shape, name=None):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial, name=name)

    @staticmethod
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    def __init__(self, save_loc=None):
        self.x = tf.placeholder(tf.float32, shape=[None, 784]) # Define input placeholder
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10]) #Define output placeholder
        if save_loc is None:
            save_loc = "saved_model/network_weights.ckpt"
        self.save_loc = save_loc

    def build(self):

        #FIRST CONVOLUTIONAL LAYER
        W_conv1 = self.weight_variable([5, 5, 1, 32], name='W_conv1') #Define the weights of 1st convolutional layer
        b_conv1 = self.bias_variable([32], name='b_conv1') #Define the biases of 1st convolutional layer

        x_image = tf.reshape(self.x, [-1,28,28,1])

        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        #SECOND CONVOLUTIONAL LAYER
        W_conv2 = self.weight_variable([5, 5, 32, 64], name='W_conv2') #Define the weights of 2nd conv. layer
        b_conv2 = self.bias_variable([64], name='b_conv2') #Define the biases of the 2nd conv. layer

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        #DENSELY CONNECTED LAYER
        W_fc1 = self.weight_variable([7 * 7 * 64, 1024], name='W_fc1')
        b_fc1 = self.bias_variable([1024], name='b_fc1')

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        #DROPOUT
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        #READOUT LAYER
        W_fc2 = self.weight_variable([1024, 10], name='W_fc2')
        b_fc2 = self.bias_variable([10], name='b_fc2')

        self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    def train(self):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(20000):
                batch = mnist.train.next_batch(50)
                if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        self.x:batch[0], self.y_: batch[1], self.keep_prob: 1.0})
                    print("step %d, training accuracy %g"%(i, train_accuracy))
                    saver.save(sess, self.save_loc)
                train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})

            print("test accuracy %g"%accuracy.eval(feed_dict={
                self.x: mnist.test.images, self.y_: mnist.test.labels, self.keep_prob: 1.0}))
    def inference(self):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        saver = tf.train.Saver()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            saver.restore(sess, self.save_loc)

            test_data, test_labels = mnist.test.images, mnist.test.labels

            feed_dict = {
                self.x: test_data,
                self.y_: test_labels,
                self.keep_prob: 0.5
            }
            pred = tf.argmax(self.y_conv,1)

            preds = sess.run(pred, feed_dict=feed_dict)

            print(preds)

    def restore(self, sess):
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        saver.restore(sess, self.save_loc)
    def inference_one(self, sess, image):

        image = image.reshape(1, 784)

        feed_dict = {
            self.x: image,
            #self.y_: test_labels,
            self.keep_prob: 0.5
        }
        pred = tf.argmax(self.y_conv,1)

        preds = sess.run(pred, feed_dict=feed_dict)

        #print(preds)
        return preds[0]

