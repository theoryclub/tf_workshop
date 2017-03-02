import tensorflow as tf
import numpy as np
import struct
import math
import os

path=os.path.dirname(os.path.realpath(__file__))

#function to read MNIST images and labels into numpy matrices
# def readMNIST(imagesFile, labelsFile, numImages):
#     image_file=open(imagesFile, "rb")
#     imageBytes=image_file.read()
#     image_file.close()    
#     labelFile=open(labelsFile, "rb")
#     labelsBytes=labelFile.read()
#     labelFile.close()    
#     images=np.zeros([numImages, 28*28], np.float32)
#     labels=np.zeros([numImages, 10], np.float32)   
#     imageOffset=16
#     labelOffset=8    
#     for imageInd in range(0, numImages):
#         labels[imageInd][struct.unpack('B', labelsBytes[labelOffset+imageInd])[0]]=1.0
#         for row in range(0, 28):
#             for col in range(0, 28):
#                 images[imageInd][28*row+col]=struct.unpack('B', imageBytes[imageOffset+28*28*imageInd+28*row+col])[0]/256.0
#     return images, labels

def getClassificationAccuracy(networkOutputs, trueLabels):
    numberCorrect=0.0
    for labelInd in range(0, len(trueLabels)):
        if trueLabels[labelInd][np.argmax(networkOutputs[labelInd], 0)]==1:
            numberCorrect=numberCorrect+1
    print('Classification Accuracy: '+str(100*(numberCorrect/len(trueLabels)))+'%')

print 'Training a neural network on the MNIST Handwriting Classification Problem'

inputs = tf.placeholder(tf.float32, ([None, 28*28])) #inputs placeholder
trueOutput = tf.placeholder(tf.float32, ([None, 10])) #correct image label placeholder

IMAGE_PIXELS=28*28
NUM_CLASSES=10
hidden1_num_neurons=25 #neurons in first layer
output_num_neurons=10 #neurons in second (output) layer. Each neuron corresponds to a digit. The classification is the order of the
#output neuron with the highest activation

#first layer weights and biases
weights1 = tf.Variable(tf.random_normal([IMAGE_PIXELS, hidden1_num_neurons]))
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
trainImages, trainLabels=readMNIST(path+'/data/MNIST_Numbers/train-images.idx3-ubyte',
                                    path+'/data/MNIST_Numbers/train-labels.idx1-ubyte',
                                    60000)

#train neural network
BATCH_SIZE=2500;
with tf.Session() as session:
    tf.initialize_all_variables().run()
    
    #train for 100 optimization steps (on all 60,000 inputs)
    for i in range(0, 40):
        shuffle=np.random.permutation(60000)
        sessLoss=0.0
        sessOutput=np.zeros([60000, 10])
        for batchInd in range(0, 60000, BATCH_SIZE):
            _, batchLoss, batchOutput=session.run([train_op, loss, output], feed_dict={inputs: trainImages[shuffle[batchInd:batchInd+BATCH_SIZE]], 
                                                                                       trueOutput: trainLabels[shuffle[batchInd:batchInd+BATCH_SIZE]]})
            sessLoss+=batchLoss
            sessOutput[batchInd:batchInd+BATCH_SIZE]=batchOutput
        print 'Epoch '+str(i)+' train loss', sessLoss/(60000/BATCH_SIZE)
        getClassificationAccuracy(sessOutput, trainLabels)
        print
    
    testImages, testLabels=readMNIST(path+'/data/MNIST_Numbers/t10k-images.idx3-ubyte',
                                        path+'/data/MNIST_Numbers/t10k-labels.idx1-ubyte',
                                        10000) 
    
    sessLoss, sessOutput=session.run([loss, output], feed_dict={inputs: testImages, trueOutput: testLabels})  
    print'test loss', sessLoss
    getClassificationAccuracy(sessOutput, testLabels)
 
