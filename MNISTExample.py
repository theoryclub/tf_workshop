import tensorflow as tf
import numpy as np

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

FEATURES=22
NUM_CLASSES=2
hidden1_num_neurons=2 #neurons in first layer
output_num_neurons=NUM_CLASSES #neurons in second (output) layer. Each neuron corresponds to a digit. The classification is the order of the
#output neuron with the highest activation

#function to read MNIST images and labels into numpy matrices
def loadData(file):
    splitLines=readToLines(file)
    voiceFeatures=np.zeros([len(splitLines)-1, FEATURES])
    labels=np.zeros([len(splitLines)-1, NUM_CLASSES])
    for dataInd in range(1, len(splitLines)):
        splitLine=splitLines[dataInd]
        voiceFeatures[dataInd-1, :16]=splitLine[1:17]
        voiceFeatures[dataInd-1, 16:]=splitLine[18:]
        labels[dataInd-1, int(splitLine[17])]=1.0
    
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
            
    shuffle=np.random.permutation(len(splitLines)-1)  
    
    return voiceFeatures[shuffle[0:int(3*(len(splitLines)-1)/4)]], labels[shuffle[0:int(3*(len(splitLines)-1)/4)]], voiceFeatures[shuffle[int(3*(len(splitLines)-1)/4):]], labels[shuffle[int(3*(len(splitLines)-1)/4):]]

trainVoiceFeatures, trainLabels, validateVoiceFeatures, validateLabels=loadData('/home/willie/workspace/TensorFlowWorkshop/data/Parkinsons/parkinsons.data')


x = tf.placeholder(tf.float32, [None, FEATURES])
W = tf.Variable(tf.zeros([FEATURES, NUM_CLASSES]))
b = tf.Variable(tf.zeros([NUM_CLASSES]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.initialize_all_variables().run()
for _ in range(100):
    crossEntropy, acc, _=sess.run([cross_entropy, accuracy, train_step], feed_dict={x: trainVoiceFeatures, y_: trainLabels})
    print crossEntropy, acc
    print(sess.run(accuracy, feed_dict={x: validateVoiceFeatures, y_: validateLabels}))