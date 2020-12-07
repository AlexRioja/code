import numpy as np 
import matplotlib.pyplot as plt

input = np.linspace(-10,10,100)

def sigmoid(x):
    val=1/(1+np.exp(-x))
    return val

output = sigmoid(input)

plt.plot(input, output)
plt.xlabel("Input")
plt.ylabel("Output")
plt.title("Sigmoid Function")
plt.show()