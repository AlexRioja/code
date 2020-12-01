import numpy as np 				# Numerical process
import scipy as sc 				# EXtras for numpi
import matplotlib.pyplot as plt # Graphics
import time 					# have some measures when training

from sklearn.datasets import make_circles 	# prepare our dataset
from IPython import display 				# System display

# DATASET

n = 500 	#Number of registers/elements of data
p = 2 		#Number of features of our data

X, Y = make_circles(n_samples=n, factor=0.4, noise=0.05)

Y = Y[:, np.newaxis] 

#plt.scatter(X[Y==0, 0], X[Y==0,1], c="red")
#plt.scatter(X[Y==1, 0], X[Y==1,1], c="blue")

#plt.axis("equal")
#plt.show()

# Creating layers of our NN 
# NEW CLASS 

class neural_layer():

	def __init__(self, number_connections, number_neurons, activation_function):

		self.activation_function = activation_function

		self.b = np.random.rand(1, number_neurons) * 2 - 1 
		self.W = np.random.rand(number_connections, number_neurons) * 2 - 1
#End of the class		

sigmoid = (lambda x: 1 / (1 + np.e ** (-x)), 	# Implementing sigmoid function
	lambda x: x * (1 - x))						# Implementing derivative function

relu = lambda x: np.maximum(0,x) 				# Implementation of RELU function

#aux = np.linspace(-10, 10, 100) #Set of number to test our activation function
#plt.plot(aux, sigmoid[0](aux))
#plt.show()
#plt.plot(aux, sigmoid[1](aux))
#plt.show()
#plt.plot(aux, relu(aux))
#plt.show()

#layer_0 = neural_layer(p, 4, sigmoid)
#layer_n = neural_layer(8, 16, relu)

topology =[p, 4, 8, 1]

def create_nn(topology, activation_function):
	
	nn = []

	for l, layer in enumerate(topology[:-1]):

		nn.append(neural_layer(topology[l], topology[l+1], activation_function))

	return nn


neural_network = create_nn(topology, sigmoid)
print(neural_network[0])

#Cost function is the mean square error
cost_func = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
	lambda Yp, Yr: (Yp - Yr))


def train(neural_network, X, Y, cost_func, learning_rate = 0.05, train = True):


	output = [(None, X)]

	# Forward pass - Step 1 of our training
	for l, layer in enumerate (neural_network):

		z = output[-1][1] @ neural_network[l].W + neural_network[l].b 
		a = neural_network[l].activation_function[0](z)

		output.append((z, a))

	# print(output[-1][1])
	# print(cost_func[0](output[-1][1], Y)) #Resulting error

	if train: # Here begin the "real" training Backwar + gradient

		deltas = [] 

		for l in reversed (range(0, len(neural_network))):

			z = output[l+1][0]
			a = output[l+1][1]

			# The first layer is not calculted in the same way than the others
			if l == len(neural_network) - 1:
				deltas.insert(0, cost_func[1](a, Y) * neural_network[l].activation_function[1](a) )

			else: 
				deltas.insert(0, deltas[0] @ _W.T * neural_network[l].activation_function[1](a) )

			# After the first iteration we will need the Weights matrix transponsed
			_W = neural_network[l].W

			#Implementation of the Gradient descent
			neural_network[l].b = neural_network[l].b - np.mean(deltas[0], axis=0, keepdims=True) * learning_rate
			neural_network[l].W = neural_network[l].W - output[l][1].T @ deltas[0] * learning_rate

	return output[-1][1]	

# Here we create our net and then we train it:
neural_net = create_nn(topology, sigmoid) 

loss = [] 

for i in range(2000): # 2000 iterations

	pY = train(neural_net, X, Y, cost_func, learning_rate=0.05)

	if i % 25 == 0: # Each 25 iterations we calculate the loss and add to the vector 

		loss.append(cost_func[0](pY, Y))

		res = 50 

		aux_x0 = np.linspace(-1.5,1.5, res) 
		aux_x1 = np.linspace(-1.5,1.5, res)

		aux_Y = np.zeros((res, res))

		for i0, x0 in enumerate(aux_x0):
			for i1, x1 in enumerate(aux_x1):
				aux_Y[i0,i1] = train(neural_net, np.array([[x0, x1]]), Y, cost_func, train=False)[0][0]

		#plot the results
		plt.pcolormesh(aux_x0, aux_x1, aux_Y, cmap="coolwarm")
		plt.axis("equal")

		plt.scatter(X[Y[:,0] == 0, 0], X[Y[:,0] == 0, 1], c="blue")
		plt.scatter(X[Y[:,0] == 1, 0], X[Y[:,0] == 1, 1], c="red")

		print(cost_func[0](pY, Y))
		plt.show()
		plt.plot(range(len(loss)), loss)
		plt.show()
		time.sleep(0.5)
