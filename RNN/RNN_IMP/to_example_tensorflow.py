import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn, rnn_cell


# training dataset generation
int2binary = {}
binary_dim = 8

largest_number = pow(2,binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

# Parameters
learning_rate = 0.001
training_iters = 200#100000
batch_size = 1

# Network Parameters
n_input = 2 # Two inputs
n_steps = 8
n_hidden = 16 # hidden layer num of features
n_classes = 1 # output 


# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
istate = tf.placeholder("float", [None, 2*n_hidden]) #state & cell => 2x n_hidden
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(_X, _istate, _weights, _biases):
	print _X, 'X'
	# input shape: (batch_size, n_steps, n_input)
	_X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
	# Reshape to prepare input to hidden activation
	print _X, 'Transpose'
	_X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input)
	print _X

	# Linear activation
	_X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

	# Define a lstm cell with tensorflow
	lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
	# Split data because rnn cell needs a list of inputs for the RNN inner loop
	_X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)

	# Get lstm cell output
	outputs, states = rnn.rnn(lstm_cell, _X, initial_state=_istate)

	# Linear activation
	# Get inner loop last output
	return tf.matmul(outputs[-1], _weights['out']) + _biases['out']

pred = RNN(x, istate, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
		# generate a simple addition problem (a + b = c)
		a_int = np.random.randint(largest_number/2) # int version
		a = int2binary[a_int] # binary encoding
		b_int = np.random.randint(largest_number/2) # int version
		b = int2binary[b_int] # binary encoding
		n = np.zeros((len(a),2))
		n[:,0] = a
		n[:,1] = b
		batch_xs = [n]

		# true answer
		c_int = a_int + b_int
		c = int2binary[c_int]
		batch_ys = [[c_int]]
		
		# Fit training using batch data
		sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
			istate: np.zeros((batch_size, 2*n_hidden))})
		if step % 100 == 0:
			# Calculate batch accuracy
			acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
			istate: np.zeros((batch_size, 2*n_hidden))})
			# Calculate batch loss
			loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,
			istate: np.zeros((batch_size, 2*n_hidden))})
			print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
		step += 1
		a_int = np.random.randint(largest_number/2) # int version
		a = int2binary[a_int] # binary encoding
		b_int = np.random.randint(largest_number/2) # int version
		b = int2binary[b_int] # binary encoding
		n = np.zeros((len(a),2))
		n[:,0] = a
		n[:,1] = b
		batch_xs = [n]

		# true answer
		c_int = a_int + b_int
		c = int2binary[c_int]
		batch_ys = [[c_int]]
		print c_int
		print sess.run(pred, feed_dict={x: batch_xs, istate: np.zeros((1, 2*n_hidden))})
		