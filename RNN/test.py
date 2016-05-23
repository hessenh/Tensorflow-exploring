import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import load_data_set

data_set = load_data_set.main()

n_input = 6
batch_size = 100
n_steps = 100
n_hidden = 100
n_classes = 10
learning_rate = 0.001
num_iterations = 1000

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input],"X")
print x
istate = tf.placeholder("float", [None, 2*n_hidden],"istate") #state & cell => 2x n_hidden
print istate
y = tf.placeholder("float", [None, n_classes],"y")
print y

weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden]),'hidden_weights'), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]),'out_weights')
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden]),'hidden_biases'),
    'out': tf.Variable(tf.random_normal([n_classes]),'out_biases')
}


# input shape: (batch_size, n_steps, n_input)
_X = tf.transpose(x, [1, 0, 2],'X_Transpose')  # permute n_steps and batch_size
# Reshape to prepare input to hidden activation
print _X
_X = tf.reshape(_X, [-1, n_input],'X_Reshape') # (n_steps*batch_size, n_input)
print _X
print weights['hidden'].get_shape()
# Linear activation
_X = tf.matmul(_X, weights['hidden']) + biases['hidden']
print _X, 'matmul'

# Define a lstm cell with tensorflow
lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

# Split data because rnn cell needs a list of inputs for the RNN inner loop
_X_split = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)
#print _X_split, 'split'


# Get lstm cell output
outputs, states = rnn.rnn(lstm_cell, _X_split, initial_state=istate)
#print outputs,'outputs'
#print states,'states'
# Linear activation
# Get inner loop last output
pred =  tf.matmul(outputs[-1], weights['out']) + biases['out']
print pred
#pred = RNN(x, istate, weights, biases)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()




#  Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    for i in range(0,num_iterations):
        batch_xs, batch_ys = data_set.next_training_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements

        batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
        print batch_xs
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                       istate: np.zeros((batch_size, 2*n_hidden))})
        if i % 100 == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
                                                istate: np.zeros((batch_size, 2*n_hidden))})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,
                                             istate: np.zeros((batch_size, 2*n_hidden))})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                  ", Training Accuracy= " + "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"
    # Calculate accuracy for 256 mnist test images
    test_len = 256
    test_data = batch_xs
    test_label = batch_ys
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                                             istate: np.zeros((batch_size, 2*n_hidden))})