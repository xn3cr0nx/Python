import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#index
'''
input > weight > hidden layer 1 (activation function) > 
> weights > hidden layer 2 (activation function) > weights > output layer

compare output to intended output > cost or loss function(cross entropy)
optimizazion function(optimizer) > minimize cost (AdamOptimizer... SGD, AdaGrad)

backpropagation

feed forward + backprop = epoch
'''

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)	#one_hot = one on and the rest is off

#10 classes, 0-9
'''
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0,0,0]
.
.
9 = [0,0,0,0,0,0,0,0,0,1]
'''


###################################Model

#free dimension
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100	#features size


x = tf.placeholder('float')#[None, 784])	#input data, 784=28*28
y = tf.placeholder('float')	#label


#model the neural network
shape = [784, n_nodes_hl1]
def neural_network_model(data):
	
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal(shape)),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
					'biases':tf.Variable(tf.random_normal([n_classes]))}

	#input_data * weights + biases
	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
	
	return output


#################################Running


def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))	#it calculate the difference between the output from the deepnet(prediction) and the data we still had (y)

	#									learning_rate = 0.01
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	hm_epochs = 10	#how many epochs, cycles feed forward + backpropagation

	#train our data
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict={x:epoch_x, y:epoch_y})
				epoch_loss += c
			print 'Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss
			
		#testing data
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print 'Accuracy', accuracy.eval({x:mnist.test.images, y:mnist.test.labels})
	



#run
train_neural_network(x)




