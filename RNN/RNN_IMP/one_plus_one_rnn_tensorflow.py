''''
From: https://github.com/yankev/tensorflow_example/blob/master/rnn_example.ipynb
'''


import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn

#Defining some hyper-params
num_units = 2       #this is the parameter for input_size in the basic LSTM cell
input_size = 2      #num_units and input_size will be the same

batch_size = 100
seq_len = 5
num_hidden = 10
num_epochs=1
learning_rate = 0.001


def gen_data(min_length=3, max_length=55, n_batch=5):

    X = np.concatenate([np.random.uniform(size=(n_batch, max_length, 1)),
                        np.zeros((n_batch, max_length, 1))],
                         axis=-1)
    y = np.zeros((n_batch,))
    # Compute masks and correct values
    for n in range(n_batch):
        # Randomly choose the sequence length
        length = np.random.randint(min_length, max_length)
        #i changed this to a constant
        #length=55

        # Zero out X after the end of the sequence
        X[n, length:, 0] = 0
        # Set the second dimension to 1 at the indices to add
        X[n, np.random.randint(length/2-1), 1] = 1
        X[n, np.random.randint(length/2, length), 1] = 1
        # Multiply and sum the dimensions of X to get the target value
        y[n] = np.sum(X[n, :, 0]*X[n, :, 1])
    # Center the inputs and outputs
    #X -= X.reshape(-1, 2).mean(axis=0)
    #y -= y.mean()
    return (X,y)



# tf Graph input
x = tf.placeholder("float", [None, seq_len, input_size])
istate = tf.placeholder("float", [None, 2*num_hidden]) #state & cell => 2x n_hidden
y = tf.placeholder("float", [None, 1])

# Define weights
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

prediction = tf.matmul(outputs[-1], weights['out']) + biases['out']                                                            #each element is a tensor of size [batch_size,num_units]

cost = tf.reduce_mean(tf.pow(prediction-y,2))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer




### Execute

with tf.Session() as sess:

    tf.initialize_all_variables().run()     #initialize all variables in the model

    for k in range(10000):

        tempX,output = gen_data(4 ,seq_len,batch_size)
      
        #print tempX
        d = {
            x: tempX,
            y: np.split(output, len(output)),
            istate: np.zeros((batch_size, 2*num_hidden))
        }
        sess.run(optimizer, feed_dict= d)

        if k % 100 == 0:

            #print sess.run(prediction, feed_dict= {x: tempX})
            tempX,output = gen_data(4 ,seq_len,100)
            output = np.split(output, len(output))
            d = {
                x: tempX,
                y: output,
                istate: np.zeros((batch_size, 2*num_hidden))
            }

            print 'Cost',np.sum(sess.run(cost, feed_dict=d))
            print 'Prediction',sess.run(prediction, feed_dict = {x: [tempX[0]]}), 'Actual', output[0][0]
       