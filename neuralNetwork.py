import numpy as np


#scipy.special for the sigmoid function expit()
import scipy.special

import matplotlib.pyplot 

#neural network class definition
class neuralNetwork:

	#initialise the neural network
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		# set number of nodes in each input, hidden, output layer
		self.inodes=inputnodes
		self.hnodes=hiddennodes
		self.onodes=outputnodes

		#link weight matrices, wih and who
		#weights inside the arrays are w_i_j, where link is from node i to node j in he next layer
		#normal distribution
		self.wih=np.random.normal(0.0, pow(self.hnodes,-0.5), (self.hnodes,self.inodes))
		self.who=np.random.normal(0.0, pow(self.onodes,-0.5), (self.onodes,self.hnodes))
		
		#learning rate
		self.lr=learningrate

		#activation function is the sigmoid function
		self.activation_function=lambda x: scipy.special.expit(x)
		pass

	#train the neural network
	def train(self, inputs_list, targets_list):
		#convert inputs list to 2d array, .T means matrix transport
		inputs=np.array(inputs_list, ndmin=2).T
		targets=np.array(targets_list, ndmin=2).T

		#calculate signals into hidden layer
		hidden_inputs=np.dot(self.wih, inputs)
		#calculate the signals emerging from hidden layer
		hidden_outputs=self.activation_function(hidden_inputs)

		final_inputs=np.dot(self.who,hidden_outputs)

		#calculate the signal emerging from hidden layer
		final_outputs=self.activation_function(final_inputs)

		#error is the (target-actual)
		output_errors=targets-final_outputs

		#hidden layer error is the output_errors, split by weights, recombined at hidden nodes
		hidden_errors=np.dot(self.who.T, output_errors)

		#update the weights for the links between the hidden and output layers
		self.who+=self.lr*np.dot(output_errors*final_outputs*(1-final_outputs), np.transpose(hidden_outputs))
		pass

		#update the weights for the links between the hidden and input layers
		self.wih+=self.lr*np.dot(hidden_errors*hidden_outputs*(1-hidden_outputs), np.transpose(inputs))
		pass

	#query the neural network
	def query(self, inputs_list):
		hidden_inputs=np.dot(self.wih,inputs_list)

		#calculate the signal emerging from hidden layer
		hidden_outputs=self.activation_function(hidden_inputs)

		final_inputs=np.dot(self.who,hidden_outputs)

		#calculate the signal emerging from hidden layer
		final_outputs=self.activation_function(final_inputs)
		
		return final_outputs



#number of input, hidden and output nodes
input_nodes=784
hidden_nodes=100
output_nodes=10

#learning rate 
learning_rate=0.2

#create an instance of neural network
n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

#load the mnist training data CSV file into a list
training_data_file=open("mnist_train.csv", 'r')
training_data_list=training_data_file.readlines()
training_data_file.close()

#train the neural network

#epochs is the number of times the training data set is used for training
epochs=2

#for e in range(epochs):
#go through all records in the training data set for record in the training_data_file
for record in training_data_list:
        all_values=record.split(',')
        inputs=(np.asfarray(all_values[1:])/255.0*0.99)+0.01
        targets=np.zeros(output_nodes)+0.01

        targets[int(all_values[0])]=0.99
        n.train(inputs,targets)
pass
#pass

#load the mnsit test data CSV file into a list
test_data_file=open("mnist_test.csv", 'r')
test_data_list=test_data_file.readlines()
test_data_file.close()


#scorecard for how well the network performs, initially empty
scorecard=[]

for record in test_data_list:
        all_values=record.split(',')
        correct_label=int(all_values[0])
        #print(correct_label,"correct label")
        inputs=np.asfarray(all_values[1:])/255.0*0.99+0.01
        outputs=n.query(inputs)
        label=np.argmax(outputs)
        #print(label,"network's answer")
        if(label==correct_label):
                scorecard.append(1)
        else:
                scorecard.append(0)
                pass
        pass
#calculate the performance score
scorecard_array=np.asarray(scorecard)
print("performance=", scorecard_array.sum()/scorecard_array.size)
import cv2
img_array=cv2.imread("0.png")
img_array=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
img_array=img_array.reshape(784)
img_data=255.0-img_array
img_data=(img_data/255.0*0.99)+0.01
outputs=n.query(img_data)
print(np.argmax(outputs))


import cv2
img_array=cv2.imread("1.png")
img_array=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
img_array=img_array.reshape(784)
img_data=255.0-img_array
img_data=(img_data/255.0*0.99)+0.01
outputs=n.query(img_data)
print(np.argmax(outputs))

import cv2
img_array=cv2.imread("2.png")
img_array=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
img_array=img_array.reshape(784)
img_data=255.0-img_array
img_data=(img_data/255.0*0.99)+0.01
outputs=n.query(img_data)
print(np.argmax(outputs))

img_array=cv2.imread("3.png")
img_array=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
img_array=img_array.reshape(784)
img_data=255.0-img_array
img_data=(img_data/255.0*0.99)+0.01
outputs=n.query(img_data)
print(np.argmax(outputs))

img_array=cv2.imread("4.png")
img_array=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
img_array=img_array.reshape(784)
img_data=255.0-img_array
img_data=(img_data/255.0*0.99)+0.01
outputs=n.query(img_data)
print(np.argmax(outputs))

img_array=cv2.imread("5.png")
img_array=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
img_array=img_array.reshape(784)
img_data=255.0-img_array
img_data=(img_data/255.0*0.99)+0.01
outputs=n.query(img_data)
print(np.argmax(outputs))

img_array=cv2.imread("6.png")
img_array=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
img_array=img_array.reshape(784)
img_data=255.0-img_array
img_data=(img_data/255.0*0.99)+0.01
outputs=n.query(img_data)
print(np.argmax(outputs))

img_array=cv2.imread("7.png")
img_array=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
img_array=img_array.reshape(784)
img_data=255.0-img_array
img_data=(img_data/255.0*0.99)+0.01
outputs=n.query(img_data)
print(np.argmax(outputs))

img_array=cv2.imread("8.png")
img_array=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
img_array=img_array.reshape(784)
img_data=255.0-img_array
img_data=(img_data/255.0*0.99)+0.01
outputs=n.query(img_data)
print(np.argmax(outputs))

img_array=cv2.imread("9.png")
img_array=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
img_array=img_array.reshape(784)
img_data=255.0-img_array
img_data=(img_data/255.0*0.99)+0.01
outputs=n.query(img_data)
print(np.argmax(outputs))














