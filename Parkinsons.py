import tensorflow as tf
import numpy as np
import os

NUM_FEATURES=22
NUM_CLASSES=2
FEATURE_INDS=range(1, 17)+range(18, 24)
LABEL_IND=17
DATA_LOCATION=os.getcwd()+os.sep+'data'+os.sep+'Parkinsons'+os.sep+'parkinsons.data'

#function to read voice features and labels into numpy matrices
def loadData(file):
    #read file
    csvFile=open(file)
    lines=csvFile.read().splitlines()
    csvFile.close()
    splitLines=[]
    for line in lines:
        splitLines+=[line.split(',')]
       
    #convert file data into feature and label numpy arrays 
    voiceFeatures=np.zeros([len(splitLines)-1, NUM_FEATURES])
    labels=np.zeros([len(splitLines)-1, NUM_CLASSES])
    for dataInd in range(1, len(splitLines)):
        splitLine=splitLines[dataInd]
        voiceFeatures[dataInd-1, :]=[splitLine[ind] for ind in FEATURE_INDS]
        labels[dataInd-1, int(splitLine[LABEL_IND])]=1.0 #one-hot encoding: one column per label class, column of correct class is set to 1
    
    #normalize feature columns
    for col in range(0, len(voiceFeatures[0])):
        max=0.0
        min=float('inf')
        for row in range(0, len(voiceFeatures)):
            if(voiceFeatures[row,col]>max):
                max=voiceFeatures[row,col]
            if(voiceFeatures[row,col]<min):
                min=voiceFeatures[row,col]
        for row in range(0, len(voiceFeatures)):
            voiceFeatures[row,col]+=min
            voiceFeatures[row,col]/=(max-min)  
    
    #select random data points for training and validation
    shuffle=np.random.permutation(len(splitLines)-1)  
    return voiceFeatures[shuffle[0:int(3*(len(splitLines)-1)/4)]], labels[shuffle[0:int(3*(len(splitLines)-1)/4)]], voiceFeatures[shuffle[int(3*(len(splitLines)-1)/4):]], labels[shuffle[int(3*(len(splitLines)-1)/4):]]

#get Parkinson's data
trainVoiceFeatures, trainLabels, validateVoiceFeatures, validateLabels=loadData(DATA_LOCATION)


features = tf.placeholder(tf.float32, [None, NUM_FEATURES]) #feature input
weights = tf.Variable(tf.zeros([NUM_FEATURES, NUM_CLASSES]))
biases = tf.Variable(tf.zeros([NUM_CLASSES]))

pred_labels = tf.nn.softmax(tf.matmul(features, weights) + biases) #compute neural network layer to get predicted parkinson's classifications
                                                                   #=softmax(features*weights+biases)
                                                                    
true_labels = tf.placeholder(tf.float32, [None, NUM_CLASSES]) #correct parkinson's classifications

cross_entropy = tf.reduce_mean(-tf.reduce_sum(true_labels * tf.log(pred_labels), reduction_indices=[1])) #error in predictions
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy) #funtion to minimize error

#compute neural network accuracy
correct_prediction = tf.equal(tf.argmax(pred_labels,1), tf.argmax(true_labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#intialize tensorflow
sess = tf.InteractiveSession()
tf.initialize_all_variables().run()

#train neural network for 100 epochs
for _ in range(300):
    #train call
    crossEntropy, acc, _=sess.run(
        [cross_entropy, accuracy, train_step], 
        feed_dict={features: trainVoiceFeatures, true_labels: trainLabels})
    print( 'crossEntropy', crossEntropy, 'acc', acc)
    
    #evaluate network accuracy on validation data
    print('validation acc ', sess.run(
        accuracy, #return the 'accuracy' variable
        feed_dict={features: validateVoiceFeatures, true_labels: validateLabels})) #input 'validateVoiceFeatures' as 'features'
                                                                    #and 'validateLabels' as 'true_labels' to the computation graph
    