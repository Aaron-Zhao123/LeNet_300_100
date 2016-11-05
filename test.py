import tensorflow as tf
import numpy as np
import input_data
import sys
import os.path

sys.dont_write_bytecode = True

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
OUTPUT_SIZE = 10
n_hidden_1 = 300
n_hidden_2 = 100

# initialize a weight variable
def createWeight(shape):
    initial = tf.truncated_normal(shape, stddev = 1)
    # initial = tf.random_normal(shape, stddev = 1)
    return tf.Variable(initial)

# initialize a bias variable
def createBias(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary('stddev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)
def LeNet_300_100(x):
    n_input = IMAGE_SIZE
    w_fc1 = createWeight([n_input, n_hidden_1])
    # w_fc1 = tf.Variable(tf.random_normal([n_input, n_hidden_1],stddev = 1))
    # w_fc1_hist = tf.histogram_summary("weights fc1", w_fc1)
    b_fc1 = createBias([n_hidden_1])
    w_fc2 = createWeight([n_hidden_1, n_hidden_2])
    b_fc2 = createBias([n_hidden_2])
    w_out = createWeight([n_hidden_2, OUTPUT_SIZE])
    b_out = createBias([OUTPUT_SIZE])
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x,w_fc1), b_fc1))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1,w_fc2), b_fc2))
    layer_out = tf.nn.relu(tf.add(tf.matmul(layer_2,w_out), b_out))
    return layer_out, w_fc2

def main():
    mnist = input_data.read_data_sets("MNIST.data/", one_hot = True)
    # create model
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE])
    y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])
    sess = tf.InteractiveSession()

    y_fc, w_fc2 = LeNet_300_100(x)

    # train and evaluation
    with tf.name_scope('cross_entropy'):
    # this cap is necessary to prevent 0s
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_fc, y))
        tf.scalar_summary('cross entropy', cross_entropy)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_fc,1), tf.argmax(y,1))

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)

    merged = tf.merge_all_summaries()

    train_writer = tf.train.SummaryWriter('log',sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    if (os.path.isfile("tmp/model.ckpt")):
        saver.restore(sess,"tmp/model.ckpt")
        print("found model, restored")

    w_init = w_fc2.eval(sess)
    print w_init
    for i in range(5000000):
        batch = mnist.train.next_batch(50)
        if i%100== 0:
    	    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y: batch[1]})
            train_cross_entropy = cross_entropy.eval(feed_dict={x:batch[0], y: batch[1]})
            w_val = w_fc2.eval(sess)
            print np.equal(w_init, w_val)
            # print np.isnan(np.min(w_val))
            # print w_val
            # w_val = w_fc2.eval(feed_dict={x:batch[0], y: batch[1]})
	    with open('log/data.txt',"a") as output_file:
    		output_file.write("{},{} {}\n".format(i,train_accuracy, train_cross_entropy))
        if i%10000== 0 and i != 0:
            saver.save(sess, "tmp/model.ckpt")
        print("step %d, training accuracy %g"%(i, train_accuracy))
        summary,_ = sess.run([merged, train_step],feed_dict={x: batch[0], y: batch[1]})
        train_writer.add_summary(summary,i)
    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y: mnist.test.labels}))

if __name__ == '__main__':
    main()
