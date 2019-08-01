import tensorflow as tf

from matplotlib import pyplot as plt
from random import randint
num = randint(0, mnist.test.images.shape[0])
img = mnist.test.images[num]


with tf.Session() as sess:
    saver.restore(sess, "./my_mnist_model")
    classification = sess.run(tf.argmax(y, 1), feed_dict={x: [img]})
