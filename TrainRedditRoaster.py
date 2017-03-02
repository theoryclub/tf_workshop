from tensorflow.contrib.layers import fully_connected
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
import numpy as np
import os
import operator
from PIL import Image
from resizeimage import resizeimage
import PIL
import random
import math
from gensim.models import word2vec
import logging
from PIL.WmfImagePlugin import word
from sklearn.metrics import log_loss
 

maxTransferRate=4.8648958*10000000000

numSymbols=500
num_steps=10
numberInputs=3
numLayers=4
lstm_size=5

imageWidth=600
imageHeight=768

nnImageWidth=25
nnImageHeight=32

wordMapping={}
unscaledImages=[]
model=''

colScales={}
colOffsets={}

def scaleTrain(dataName, data):
    colScales[dataName]={}
    colOffsets[dataName]={}
    for col in range(0, len(data[0, 0])):
        colMin=np.min(data[:, :, col])
        colMax=np.max(data[:, :, col])
        colScales[dataName][col]=colMax-colMin
        colOffsets[dataName][col]=-colMin
        
def scale(dataName, data):
    for col in range(0, len(data[0, 0])):
        data[:, :, col]+=colOffsets[dataName][col]
        data[:, :, col]/=colScales[dataName][col]
    return data
        
def scaleSingle(dataName, data):
    for col in range(0, len(data)):
        data[col]+=colOffsets[dataName][col]
        data[col]/=colScales[dataName][col]
    return data

def unmap(mappedWord):
    global wordMapping
    minDist=float("inf")
    minWord='ERROR'
    for word in wordMapping.keys():
        dist=np.mean((wordMapping[word]-mappedWord)*(wordMapping[word]-mappedWord))
        if dist<minDist:
            minDist=dist
            minWord=word
    return minWord

def crossEntropy(vec1, vec2):
    return -vec1*np.log(vec2)

def readToLines(file):
    csvFile=open(file)
    lines=csvFile.read().splitlines()
    return lines

def loadData(file):
    global wordMapping
    global unscaledImages
    global model
    numRoasts=0.0
    wordDict={}
    sentances=[]
    for fileName in os.listdir(file):
        if fileName.startswith('roasts'):
            roasts=readToLines(file+fileName)
            for roastInd in range(0, min(3, len(roasts))):
                sentances+=roasts[roastInd]+' '
                roastWords=roasts[roastInd].split(' ')
                sentances+=[roastWords]
                for word in roastWords:
                    if not wordDict.has_key(word):
                        wordDict[word]=0
                    wordDict[word]+=1   
            numRoasts+=min(3, len(roasts))
        elif fileName.startswith('desc'):
            desc=readToLines(file+fileName)
            #sentances+=desc[0]+' '
            word=desc[0].split(' ')
            sentances+=[word]
            for word in roastWords:
                if not wordDict.has_key(word):
                    wordDict[word]=0
                wordDict[word]+=1
                
    wordsFreqList=sorted(wordDict.items(), key=operator.itemgetter(1))
    wordsFreqList=wordsFreqList[::-1]
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model=word2vec.Word2Vec(sentances[0:5000], numSymbols, hs=1)
    
    for sentance in sentances:
        for word in sentance:
            try:
                wordMapping[word]=model[word]
            except KeyError:
                u=0
    
#     descriptions=np.zeros([int(7*numRoasts/8), num_steps, numSymbols], np.float32)
#     images=np.zeros([int(7*numRoasts/8), nnImageWidth, nnImageHeight, 3], np.float32)
#     roasts=np.zeros([int(7*numRoasts/8), num_steps, numSymbols], np.float32)
#       
#     valDescriptions=np.zeros([int(math.ceil(numRoasts/8)), num_steps, numSymbols], np.float32)
#     valImages=np.zeros([int(math.ceil(numRoasts/8)), nnImageWidth, nnImageHeight, 3], np.float32)
#     valRoasts=np.zeros([int(math.ceil(numRoasts/8)), num_steps, numSymbols], np.float32)
    
    
    descriptions=np.zeros([10, num_steps, numSymbols], np.float32)
    images=np.zeros([10, nnImageWidth, nnImageHeight, 3], np.float32)
    roasts=np.zeros([10, num_steps, numSymbols], np.float32)
      
    valDescriptions=np.zeros([10, num_steps, numSymbols], np.float32)
    valImages=np.zeros([int(numRoasts/8), nnImageWidth, nnImageHeight, 3], np.float32)
    valRoasts=np.zeros([int(numRoasts/8), num_steps, numSymbols], np.float32)
    
    
    dataInd=0
    for fileName in os.listdir(file):
        if fileName.startswith('roasts'):
            
            roast=readToLines(file+fileName)
            for roastInd in range(0, min(3, len(roast))):
                roastWords=roast[roastInd].split(' ')
                for wordInd in range(0, min(num_steps, len(roastWords))):
                    if wordMapping.has_key(roastWords[wordInd]):
                        if dataInd<int(7*numRoasts/8):
                            roasts[dataInd, wordInd, :]=wordMapping[roastWords[wordInd]]
                        else:
                            valRoasts[dataInd-int(7*numRoasts/8), wordInd, :]=wordMapping[roastWords[wordInd]]
                    else:
                        if dataInd<int(7*numRoasts/8):
                            roasts[dataInd, wordInd, numSymbols-1]=1.0
                        else:
                            valRoasts[dataInd-int(7*numRoasts/8), wordInd, numSymbols-1]=1.0
            
                roastEndInd=fileName.index('roasts')+len('roasts')
                pageNumEndInd=fileName.index('_')
                pageNum=fileName[roastEndInd:pageNumEndInd]
                postNumEndInd=fileName.index('.')
                postNum=fileName[pageNumEndInd+1:postNumEndInd]
                
                imageFile=open(file+'image'+pageNum+'_'+postNum+'.jpeg', 'r+b')
                image=Image.open(imageFile)
                unscaledImages+=[image]
                image=resizeimage.resize_cover(image, [imageWidth, imageHeight], validate=False)
                image=image.resize((nnImageWidth, nnImageHeight),  PIL.Image.ANTIALIAS)
                #image.show()
                pixels=image.load()
                for x in range(0, nnImageWidth):
                    for y in range(0, nnImageHeight):
                        for rgbInd in range(0, 3):
                            if dataInd<int(7*numRoasts/8):
                                i=pixels[x, y]
                                images[dataInd, x, y, rgbInd]=pixels[x, y][rgbInd]/255.0
                            else:
                                valImages[dataInd-int(7*numRoasts/8), x, y, rgbInd]=pixels[x, y][rgbInd]/255.0
                
                description=readToLines(file+'desc'+pageNum+'_'+postNum+'.txt')[0]
                descriptionWords=description.split(' ')
                for wordInd in range(0, min(num_steps, len(descriptionWords))):
                    if wordMapping.has_key(descriptionWords[wordInd]):
                        if dataInd<int(7*numRoasts/8):
                            descriptions[dataInd, num_steps-1-wordInd]=wordMapping[descriptionWords[wordInd]]
                        else:
                            valDescriptions[dataInd-int(7*numRoasts/8), num_steps-1-wordInd]=wordMapping[descriptionWords[wordInd]]       
                dataInd+=1
                imageFile.close()
        
                if dataInd>=10:
                    break;
            if dataInd>=10:
                break;
     
    scaleTrain('descriptions', descriptions)
    scaleTrain('roasts', roasts)
    
    scale('descriptions', descriptions)
    scale('roasts', roasts)
    scale('descriptions', valDescriptions)
    scale('roasts', valRoasts)
    for word in  wordMapping.keys():
        scaleSingle('roasts', wordMapping[word])
    
    return [descriptions, images, roasts, valDescriptions, valImages, valRoasts]
            

def permuteInPlace(matrix, permutation):
    for rowInd in range(0, len(permutation)):
        temp=matrix[rowInd]
        matrix[rowInd]=matrix[permutation[rowInd]]
        matrix[permutation[rowInd]]=temp
    return matrix

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1], padding='SAME')
    
def unmapOneHotWords(onHotWords):
    reversedWords=''
    for oneHotWord in onHotWords:
        reversedWords+=unmap(oneHotWord)+' '
    return reversedWords

descriptions, imagesData, roastsData, valDescriptions, valImages, valRoasts=loadData('/home/willie/workspace/TensorFlowWorkshop/data/RedditRoasts/')


 
desc = tf.placeholder(tf.float32, [None, num_steps, numSymbols], name='descriptions')
images=tf.placeholder(tf.float32, [None, nnImageWidth, nnImageHeight, 3], name='images')
roasts=tf.placeholder(tf.float32, [None, num_steps, numSymbols], name='roasts')
zeroWords=tf.placeholder(tf.float32, [None, num_steps, numSymbols], name='zeroWords')

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(images, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

initialState=fully_connected(tf.reshape(h_pool2, [-1, 3584]), 4*numSymbols)
hiddenState1=fully_connected(initialState, numSymbols)

cell = tf.nn.rnn_cell.GRUCell(numSymbols)
stacked_cell = rnn_cell.MultiRNNCell([cell] * numLayers)

# rnnOutputs=[]
# #rnnState=tf.zeros([numSymbols], tf.float32)
# rnnState=(hiddenState1, hiddenState1, hiddenState1)
# rnnOutputStep=hiddenState1
# #rnnOutputStep, rnnState=stacked_cell(zeroWords, rnnState)
# #rnnOutputs.append(rnnOutputStep)
# with tf.variable_scope("myrnn") as scope:
#     for i in range(0, num_steps):
#         if i > 0:
#             scope.reuse_variables()
#         rnnOutputStep, rnnState=stacked_cell(rnnOutputStep, rnnState)
#         rnnOutputs.append(rnnOutputStep)
#         
# rnnOutput = tf.reshape(tf.pack(rnnOutputs), [-1, num_steps, numSymbols])
rnnOutput, state=tf.nn.dynamic_rnn(stacked_cell, zeroWords, dtype=tf.float32, initial_state=(hiddenState1, hiddenState1, hiddenState1, hiddenState1))

rnnOutput_flattened=tf.reshape(rnnOutput, [-1, numSymbols])
roastsReshape=tf.reshape(roasts, [-1, numSymbols])
#loss=tf.reduce_mean(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets=roastsReshape, logits=rnnOutput_flattened), 0), 0)
loss=tf.reduce_mean((roastsReshape-rnnOutput_flattened)*(roastsReshape-rnnOutput_flattened))
optimizer=tf.train.RMSPropOptimizer(0.001, epsilon=1e-08)
rnnFinalOutput=tf.nn.sigmoid(rnnOutput)
#optimizer=tf.train.GradientDescentOptimizer(0.01)
global_step=tf.Variable(0, name='global_step', trainable=False)
train_op=optimizer.minimize(loss, global_step=global_step)

BATCH_SIZE=100        
with tf.Session() as session:
    tf.initialize_all_variables().run()
    saver=tf.train.Saver()
    for i in range(0, 4000):      
        sessLoss=0.0
        for batchInd in range(0, len(descriptions), BATCH_SIZE):
            batchEndInd=min(len(descriptions), batchInd+BATCH_SIZE)
            sequenceLengths=np.full([batchEndInd-batchInd, 1], num_steps)
            zeros=np.zeros([batchEndInd-batchInd, num_steps, numSymbols])
            zeros[:, 1:num_steps, :]=roastsData[batchInd:batchEndInd, 0:num_steps-1, :]
            batchLoss, _=session.run([loss, train_op], feed_dict={desc: descriptions[batchInd:batchEndInd], 
                                                                  images: imagesData[batchInd:batchEndInd],
                                                                  roasts: roastsData[batchInd:batchEndInd],
                                                                  zeroWords: zeros})#
            sessLoss+=batchLoss*(batchEndInd-batchInd)
        print 'Epoch '+str(i)+' train loss', sessLoss/float(len(descriptions))
        
        if i%10==0:
            zeros=np.zeros([1, num_steps, numSymbols])
            output=np.zeros([num_steps, numSymbols])
            dispInd=int(random.uniform(0, len(descriptions)))
            hiddenState=session.run([hiddenState1], feed_dict={desc: descriptions[dispInd:dispInd+1:, :, :], 
                                                                  images: imagesData[dispInd:dispInd+1, :, :],
                                                                  roasts: roastsData[dispInd:dispInd+1, :, :],
                                                                  zeroWords: zeros})
            hiddenState, output[0]=session.run([state, rnnOutput], feed_dict={hiddenState1: hiddenState[0],
                                                                  zeroWords: output[0, :]})
            for step in range(0, num_steps):
                hiddenState, output[step]=session.run([state, rnnOutput], feed_dict={hiddenState1: hiddenState,
                                                                                     zeroWords: output[step-1]})
                
            print 'description: '+unmapOneHotWords(descriptions[dispInd])
            print 'roast: '+unmapOneHotWords(output)
            print 'desired roast: '+unmapOneHotWords(roastsData[dispInd])
            sessLoss=0.0
#             sessOutput=np.zeros([len(valDescriptions), num_steps, numSymbols])
#             for batchInd in range(0, len(valDescriptions), BATCH_SIZE):
#                 batchEndInd=min(len(valDescriptions), batchInd+BATCH_SIZE)
#                 batchLoss, batchOutput=session.run([loss, rnnOutput], feed_dict={desc: valDescriptions[batchInd:batchEndInd], 
#                                                                       images: valImages[batchInd:batchEndInd],
#                                                                       roasts: valRoasts[batchInd:batchEndInd],
#                                                                       zeroWords: np.zeros([batchEndInd-batchInd, num_steps, numSymbols])})
#                 
#                 sessLoss+=batchLoss*(batchEndInd-batchInd)
#                 sessOutput[batchInd:batchInd+BATCH_SIZE]=batchOutput
#                 
#             print 'Validation loss ', sessLoss/float(len(valDescriptions))
#             dispInd=int(random.uniform(0, len(valDescriptions)))
#             unscaledImages[len(descriptions)+dispInd].show()
#             print 'description: '+unmapOneHotWords(valDescriptions[dispInd])
#             print 'roast: '+unmapOneHotWords(sessOutput[dispInd])

#             sessOutput=np.zeros([len(descriptions), num_steps, numSymbols])
#             for batchInd in range(0, len(descriptions), BATCH_SIZE):
#                 batchEndInd=min(len(descriptions), batchInd+BATCH_SIZE)
#                 zeros=np.zeros([batchEndInd-batchInd, num_steps, numSymbols])
#                 zeros[:, 1:num_steps, :]=roastsData[batchInd:batchEndInd, 0:num_steps-1, :]
#                 batchLoss, batchOutput=session.run([loss, rnnOutput], feed_dict={desc: descriptions[batchInd:batchEndInd], 
#                                                                       images: imagesData[batchInd:batchEndInd],
#                                                                       roasts: roastsData[batchInd:batchEndInd],
#                                                                       zeroWords: zeros})
#                  
#                 sessLoss+=batchLoss*(batchEndInd-batchInd)
#                 sessOutput[batchInd:batchInd+BATCH_SIZE]=batchOutput
#                  
#             print 'Validation loss ', sessLoss/float(len(descriptions))
#             dispInd=int(random.uniform(0, len(descriptions)))
#             #unscaledImages[dispInd].show()
#             print 'description: '+unmapOneHotWords(descriptions[dispInd])
#             print 'roast: '+unmapOneHotWords(sessOutput[dispInd])
#             print 'desired roast: '+unmapOneHotWords(roastsData[dispInd])
            
            #save_path = saver.save(session, '/home/willie/workspace/TensorFlowWorkshop/data/SavedNetworks'+str(i))
            #print("Model saved in file: %s" % save_path)
            