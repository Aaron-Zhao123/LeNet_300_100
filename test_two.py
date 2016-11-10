'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

# Import MNIST data
import input_data
import os.path
import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 10000
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 300# 1st layer number of features
n_hidden_2 = 100# 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

def main():
    mnist = input_data.read_data_sets("MNIST.data/", one_hot = True)
    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

    # Define loss and optimizer

    with tf.name_scope('cross_entropy'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        tf.scalar_summary('cross entropy', cost)

    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)

    merged = tf.merge_all_summaries()
    saver = tf.train.Saver()

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # restore model if exists
        if (os.path.isfile("tmp/model.ckpt")):
            saver.restore(sess, "tmp/model.ckpt")
            print ("model found and restored")


        # Training cycle
        training_cnt = 0
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y})
                training_cnt = training_cnt + 1
                train_accuracy = accuracy.eval(feed_dict={x:batch_x, y: batch_y})
                with open('log/data.txt',"a") as output_file:
            		output_file.write("{},{},{}\n".format(training_cnt,train_accuracy, c))
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
		saver.save(sess, "tmp/model.ckpt")
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
if __name__ == '__main__':
    main()
