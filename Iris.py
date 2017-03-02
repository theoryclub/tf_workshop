import tensorflow as tf
import numpy as np
import struct
import math
import os

path=os.path.dirname(os.path.realpath(__file__))

def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0: 
       return v
    return v/norm

def readToLines(file):
    csvFile=open(file)
    lines=csvFile.read().splitlines()
    csvFile.close()
    splitLines=[]
    for line in lines:
        splitLines+=[line.split(',')]
    return splitLines

FEATURES=4
NUM_CLASSES=3
hidden1_num_neurons=2 #neurons in first layer
output_num_neurons=NUM_CLASSES #neurons in second (output) layer. Each neuron corresponds to a digit. The classification is the order of the
#output neuron with the highest activation

#function to read MNIST images and labels into numpy matrices
def loadData(file):
    splitLines=readToLines(file)
    global FEATURES
    vocab = list(set(splitLines))
    features=np.zeros([len(splitLines)-1, FEATURES])
    labels=np.zeros([len(splitLines)-1, NUM_CLASSES])
    for dataInd in range(0, len(splitLines)):
        splitLine=splitLines[dataInd]
        features[dataInd, :]=splitLine[:4]
        labels[dataInd, int(splitLine[4])]=1.0
        
    for ind in range(0, len(features[0])):
        features[:, ind]=normalize(features[:, ind])
    
    return features[0:int(3*(len(splitLines)-1)/4)], labels[0:int(3*(len(splitLines)-1)/4)], features[int(3*(len(splitLines)-1)/4):], labels[int(3*(len(splitLines)-1)/4):]

def getClassificationAccuracy(networkOutputs, trueLabels):
    numberCorrect=0.0
    for labelInd in range(0, len(trueLabels)):
        if trueLabels[labelInd][np.argmax(networkOutputs[labelInd], 0)]==1:
            numberCorrect=numberCorrect+1
    print('Classification Accuracy: '+str(100*(numberCorrect/len(trueLabels)))+'%')

print('Training a neural network on the MNIST Handwriting Classification Problem')

inputs = tf.placeholder(tf.float32, ([None, FEATURES])) #inputs placeholder
trueOutput = tf.placeholder(tf.float32, ([None, NUM_CLASSES])) #correct image label placeholder

#first layer weights and biases
weights1 = tf.Variable(tf.random_normal([FEATURES, hidden1_num_neurons]))
biases1 = tf.Variable(tf.zeros([hidden1_num_neurons]))
hidden1 = tf.nn.sigmoid(tf.matmul(inputs, weights1) + biases1)

#second layer weights and biases
weights2 = tf.Variable(tf.random_normal([hidden1_num_neurons, output_num_neurons]))
biases2 = tf.Variable(tf.zeros([output_num_neurons]))
output = tf.nn.softmax(tf.matmul(hidden1, weights2) + biases2)

#loss function: mean squared error
loss=tf.reduce_mean(tf.square(tf.subtract(output, trueOutput)))

#specify optimization operation ('train op')
optimizer = tf.train.AdamOptimizer()
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)

#read MNIST images and tabels
trainImages, trainLabels, valImages, valLabels=loadData('/home/willie/workspace/TensorFlowWorkshop/data/iris.csv')

#train neural network
BATCH_SIZE=2500;
with tf.Session() as session:
    tf.initialize_all_variables().run()
    
    #train for 100 optimization steps (on all 60,000 inputs)
    for i in range(0, 40):
        shuffle=np.random.permutation(len(trainImages))
        sessLoss=0.0
        sessOutput=np.zeros([len(trainImages), 10])
        for batchInd in range(0, len(trainImages), BATCH_SIZE):
            _, batchLoss, batchOutput=session.run([train_op, loss, output], feed_dict={inputs: trainImages[shuffle[batchInd:batchInd+BATCH_SIZE]], 
                                                                                       trueOutput: trainLabels[shuffle[batchInd:batchInd+BATCH_SIZE]]})
            sessLoss+=batchLoss
            sessOutput[batchInd:batchInd+BATCH_SIZE]=batchOutput
        print('Epoch '+str(i)+' train loss', sessLoss)
        getClassificationAccuracy(sessOutput, trainLabels)
        print()

    sessLoss, sessOutput=session.run([loss, output], feed_dict={inputs: valImages, trueOutput: valLabels})  
    print('test loss', sessLoss)
    getClassificationAccuracy(sessOutput, valLabels)
 
