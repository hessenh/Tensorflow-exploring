'''
RNN for seq acc data
One output


'''


import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn
import load_data_stream


data_set = load_data_stream.main()

input_size = 6
output_size = 1
seq_len = 50

batch_size = 100
num_hidden = 3
learning_rate = 0.001


x = tf.placeholder("float", [None, seq_len, input_size])
istate = tf.placeholder("float", [None, 2*num_hidden]) #state & cell => 2x n_hidden
y = tf.placeholder("float", [None, output_size])



#def RNN(x, input_size, num_hidden):
weights = {
'hidden': tf.Variable(tf.random_normal([input_size, num_hidden])), # Hidden layer weights
'out': tf.Variable(tf.random_normal([num_hidden, 1]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([num_hidden])),
    'out': tf.Variable(tf.random_normal([1]))
}

X_t = tf.transpose(x, [1, 0, 2])  # permute n_steps and batch_size
# Reshape to prepare input to hidden activation
X_r = tf.reshape(X_t, [-1, input_size]) # (n_steps*batch_size, n_input)

X_m = tf.matmul(X_r, weights['hidden']) + biases['hidden']

X_s = tf.split(0, seq_len, X_m) # n_steps * (batch_size, n_hidden)

lstm_cell = rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0)

outputs, states = rnn.rnn(lstm_cell, X_s, dtype=tf.float32)   #note that outputs is a list of seq_len

prediction =  tf.matmul(outputs[-1], weights['out']) + biases['out']                                                            #each element is a tensor of size [batch_size,num_units]


#prediction = RNN(x, input_size, num_hidden)

cost = tf.reduce_mean(tf.pow(prediction-y,2))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Evaluate model
correct_pred = tf.equal(tf.round(prediction), tf.round(y))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:

    tf.initialize_all_variables().run()     #initialize all variables in the model
    step = 0
    while step < 10000:
        tempX,output = data_set.next_training_batch(batch_size, seq_len)
        # Want to have last elent from each sequence
      	output = output[0::seq_len]

      	# Split data into sequences
      	tempX = np.split(tempX, batch_size)
        sess.run(optimizer, feed_dict= {x: tempX, y: output})

        if step % 1000 == 0:
			print 'Step', step
			#tempX,output = data_set.next_training_batch(batch_size, seq_len)
			
			tempX = data_set._test_x[0:seq_len*8000]
			output = data_set._test_y[0:seq_len*8000]
			# Want to have last elent from each sequence
			output = output[0::seq_len]
			# Split data into sequences
			tempX = np.split(tempX, 8000)


			print 'Accuracy',sess.run(accuracy, feed_dict = {x: tempX, y: output})

			predicted = sess.run(prediction, feed_dict = {x:[tempX[0]]})

			print 'Prediction', predicted, 'Actual', output[0]

        step +=1
