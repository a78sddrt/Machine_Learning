# -*- coding: utf-8 -*-
"""Linear Classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kqo6z0Lsg1DPw4rnPF8HG_ueZxc26HmI
"""

import tensorflow as tf
import numpy as np

num_samples_per_class=1000
negative_samples=np.random.multivariate_normal(
    mean=[0,3], cov=[[1,0.5],[0.5,1]], size=num_samples_per_class
)

positive_samples=np.random.multivariate_normal(
    mean=[3,0], cov=[[1,0.5],[0.5,1]], size=num_samples_per_class
)
inputs=np.vstack((negative_samples,positive_samples)).astype(np.float32)
targets=np.vstack((np.zeros((num_samples_per_class,1),dtype='float32'),np.ones((num_samples_per_class,1),dtype='float32')))
print(targets.shape)
print(inputs[:,0])
import matplotlib.pyplot as plt
plt.scatter(inputs[:,0],inputs[:,1], c=targets[:,0])
plt.show()

input_dim=2
output_dim=1
W=tf.Variable(initial_value=tf.random.uniform(shape=(input_dim,output_dim)))
b=tf.Variable(initial_value=tf.random.uniform(shape=(output_dim,)))
print(W)
W[0].assign([1,])
W[1].assign([-1,])
print(W)

def model(inputs):
  return tf.matmul(inputs, W)+b

def sqaure_loss(targets,prediction):
  per_sample_losses=tf.square(targets-prediction)
  return tf.reduce_mean(per_sample_losses)

learning_rate=0.1 #神秘的挑選0.01 如果不考慮b 0.01似乎是比較好的選擇

def training_step(inputs,targets):
  with tf.GradientTape() as tape:
    predictions=model(inputs)
    loss=sqaure_loss(predictions,targets)
  grad_loss_wrt_W, grad_loss_wrt_b=tape.gradient(loss, [W,b])
  W.assign_sub(grad_loss_wrt_W*learning_rate)
  b.assign_sub(grad_loss_wrt_b*learning_rate) #為什麼不考慮b就學不好了? 感覺是loss function的問題
  return loss

for step in range(20):
  loss=training_step(inputs, targets)
  print('Loss at step %d: %.4f' %(step, loss))

predictions=model(inputs)
plt.scatter(inputs[:,0],inputs[:,1], c=predictions[:,0]>0.5)
plt.show()

x=np.linspace(-1,4,100)
y=-W[0]/W[1]*x+(0.5-b)/W[1]
plt.plot(x,y,'-r')
plt.scatter(inputs[:,0],inputs[:,1], c=predictions[:,0]>0.5)