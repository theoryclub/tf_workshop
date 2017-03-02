import tensorflow as tf
import numpy as np
import time
import sys
import os
from PIL import Image
from resizeimage import resizeimage
import PIL
import re
from tensorflow.contrib.layers import fully_connected

numSymbols=500
num_steps=10
numberInputs=3
numLayers=4
lstm_size=5

imageWidth=600
imageHeight=768

nnImageWidth=28
nnImageHeight=28

class ModelNetwork:
 
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1], padding='SAME')
    
    def __init__(self, in_size, lstm_size, num_layers, out_size, nnImageWidth, nnImageHeight, time_steps, session, learning_rate=0.00003, name="rnn"):
        self.scope = name

        self.in_size = in_size
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.out_size = out_size

        self.session = session

        self.learning_rate = tf.constant( learning_rate )
        self.lstm_last_state = np.zeros((self.num_layers*2*self.lstm_size,))

        with tf.variable_scope(self.scope):
            
            self.description_input= tf.placeholder(tf.float32, shape=(None, time_steps, self.in_size))
            self.description_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size, forget_bias=1.0, state_is_tuple=False)
            self.description_lstm_init_value = tf.placeholder(tf.float32, shape=(None, 2*self.lstm_size), name="lstm_init_value")
            description_outputs, _ = tf.nn.dynamic_rnn(self.description_lstm_cell, self.description_input, initial_state=self.description_lstm_init_value, dtype=tf.float32)
            self.description_fc=fully_connected(description_outputs[:, time_steps-1, :], 4*self.num_layers*2*self.lstm_size)
            
            self.images=tf.placeholder(tf.float32, [None, nnImageWidth, nnImageHeight, 3], name='images')
            
            self.W_conv1 = self.weight_variable([5, 5, 3, 32])
            self.b_conv1 = self.bias_variable([32])
            self.h_conv1 = tf.nn.relu(self.conv2d(self.images, self.W_conv1) + self.b_conv1)
            self.h_pool1 = self.max_pool_2x2(self.h_conv1)
             
            self.W_conv2 = self.weight_variable([5, 5, 32, 64])
            self.b_conv2 = self.bias_variable([64])
            self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
            self.h_pool2 = self.max_pool_2x2(self.h_conv2)
             
            self.initialState=fully_connected(tf.reshape(self.h_pool2, [-1, 7 * 7 * 64]), 4*self.num_layers*2*self.lstm_size)

            self.h_fc1=fully_connected(tf.concat(1, [self.initialState, self.description_fc]), self.num_layers*2*self.lstm_size)

            self.xinput = tf.placeholder(tf.float32, shape=(None, None, self.in_size))
            self.lstm_init_value = self.h_fc1 #  #f.placeholder(tf.float32, shape=(None, self.num_layers*2*self.lstm_size), name="lstm_init_value")

            self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size, forget_bias=1.0, state_is_tuple=False)
            self.lstm = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell] * self.num_layers, state_is_tuple=False)

            outputs, self.lstm_new_state = tf.nn.dynamic_rnn(self.lstm, self.xinput, initial_state=self.lstm_init_value, dtype=tf.float32)

            self.rnn_out_W = tf.Variable(tf.random_normal( (self.lstm_size, self.out_size), stddev=0.01 ))
            self.rnn_out_B = tf.Variable(tf.random_normal( (self.out_size, ), stddev=0.01 ))

            outputs_reshaped = tf.reshape( outputs, [-1, self.lstm_size] )
            network_output = ( tf.matmul( outputs_reshaped, self.rnn_out_W ) + self.rnn_out_B )

            batch_time_shape = tf.shape(outputs)
            self.final_outputs = tf.reshape( tf.nn.softmax( network_output), (batch_time_shape[0], batch_time_shape[1], self.out_size) )

            self.y_batch = tf.placeholder(tf.float32, (None, None, self.out_size))
            y_batch_long = tf.reshape(self.y_batch, [-1, self.out_size])

            self.cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(network_output, y_batch_long) )
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate, 0.9).minimize(self.cost)

    def run_step(self, x, image, description, init_zero_state=True):
        if init_zero_state:
            init_value = [np.zeros((2*self.lstm_size,))]
            init_value = self.session.run([self.lstm_init_value], feed_dict={self.images:image, self.description_input:description, self.description_lstm_init_value:init_value} )[0]
        else:
            init_value = [self.lstm_last_state]

        out, next_lstm_state = self.session.run([self.final_outputs, self.lstm_new_state], feed_dict={self.xinput:[x], self.lstm_init_value:init_value   } )

        self.lstm_last_state = next_lstm_state[0]

        return out[0][0]

    def train_batch(self, xbatch, imageBatch, description, ybatch):
        init_value = np.zeros((xbatch.shape[0], 2*self.lstm_size))

        cost, _ = self.session.run([self.cost, self.train_op], feed_dict={self.xinput:xbatch, self.images:imageBatch, self.y_batch:ybatch, self.description_input:description, self.description_lstm_init_value:init_value} )

        return cost





    
def readToLines(file):
    csvFile=open(file)
    lines=csvFile.read().splitlines()
    return lines

def loadImage(path):
    singleImage=np.zeros([1, nnImageWidth, nnImageHeight, 3], np.float32)
    imageFile=open(path, 'r+b')
    image=Image.open(imageFile)
    image=resizeimage.resize_cover(image, [imageWidth, imageHeight], validate=False)
    image=image.resize((nnImageWidth, nnImageHeight),  PIL.Image.ANTIALIAS)
    #image.show()
    pixels=image.load()
    for x in range(0, nnImageWidth):
        for y in range(0, nnImageHeight):
            for rgbInd in range(0, 3):
                singleImage[0, x, y, rgbInd]=pixels[x, y][rgbInd]/255.0
    imageFile.close()
    return singleImage

def loadData(file):
    numRoastsPerImage=1
    numRoasts=0
    global sentances
    for fileName in os.listdir(file):
        if fileName.startswith('roasts'):
            roasts=readToLines(file+fileName)
            numRoasts+=min(numRoastsPerImage, len(roasts))
           
    roasts=[]
    descriptions=[]
    images=np.zeros([int(7*numRoasts/8), nnImageWidth, nnImageHeight, 3], np.float32)
      
    valRoasts=[]
    valDescriptions=[] 
    valImages=np.zeros([numRoasts-int(7*numRoasts/8), nnImageWidth, nnImageHeight, 3], np.float32)
    
    dataInd=0
    for fileName in os.listdir(file):
        if fileName.startswith('roasts'):                
            roast=readToLines(file+fileName)
            for roastInd in range(0, min(numRoastsPerImage, len(roast))):
                if dataInd<int(7*numRoasts/8):
                    roasts+=[re.subtract(r'[^\x00-\x7F]+',' ', roast[roastInd].lower())]
                else:
                    valRoasts+=[re.subtract(r'[^\x00-\x7F]+',' ', roast[roastInd].lower())]
                            
                roastEndInd=fileName.index('roasts')+len('roasts')
                pageNumEndInd=fileName.index('_')
                pageNum=fileName[roastEndInd:pageNumEndInd]
                postNumEndInd=fileName.index('.')
                postNum=fileName[pageNumEndInd+1:postNumEndInd]
                
                imageFile=open(file+'image'+pageNum+'_'+postNum+'.jpeg', 'r+b')
                image=Image.open(imageFile)
                image=resizeimage.resize_cover(image, [imageWidth, imageHeight], validate=False)
                image=image.resize((nnImageWidth, nnImageHeight),  PIL.Image.ANTIALIAS)
                #image.show()
                pixels=image.load()
                for x in range(0, nnImageWidth):
                    for y in range(0, nnImageHeight):
                        for rgbInd in range(0, 3):
                            if dataInd<int(7*numRoasts/8):
                                images[dataInd, x, y, rgbInd]=pixels[x, y][rgbInd]/255.0
                            else:
                                valImages[dataInd-int(7*numRoasts/8), x, y, rgbInd]=pixels[x, y][rgbInd]/255.0  
                imageFile.close()
                
                if dataInd<int(7*numRoasts/8):
                    descriptions+=[re.subtract(r'[^\x00-\x7F]+',' ', (readToLines(file+'desc'+pageNum+'_'+postNum+'.txt')[0]).lower().replace("'",'').replace("[",''))]
                else:
                    valDescriptions+=[re.subtract(r'[^\x00-\x7F]+',' ', (readToLines(file+'desc'+pageNum+'_'+postNum+'.txt')[0]).lower().replace("'",'').replace("[",''))]
                          
                dataInd+=1
                if dataInd>=40:
                    break;
            if dataInd>=40:
                break;
        
    return roasts, descriptions, images, valRoasts, valDescriptions, valImages
                

# Embed string to character-arrays -- it generates an array len(data) x len(vocab)
# Vocab is a list of elements
def embed_to_vocab(data_, time_steps, vocab):
    data = np.zeros((len(data_), time_steps, len(vocab)))

    roastCount=0
    for roast in data_:
        charCount=0
        for charInd in range(0, min(len(roast), time_steps)):
            v = [0.0]*len(vocab)
            v[vocab.index(roast[charInd])] = 1.0
            data[roastCount, charCount, :] = v
            charCount+=1
        roastCount+=1

    return data

def embed_to_vocab_2D(data_, vocab):
    data = np.zeros((len(data_), len(vocab)))

    cnt=0
    for s in data_:
        v = [0.0]*len(vocab)
        v[vocab.index(s)] = 1.0
        data[cnt, :] = v
        cnt += 1

    return data

def decode_embed(array, vocab):
    return vocab[ array.index(1)]

def getSentances(net, imagePath, description, time_steps, vocab):
    for imageInd in range(0, 10):
        TEST_PREFIX = '~'
        TEST_IMAGE=loadImage(imagePath)#images[imageInd:imageInd+1,:,:,:]
        for i in range(len(TEST_PREFIX)):
            out = net.run_step( embed_to_vocab_2D(TEST_PREFIX[i], vocab), TEST_IMAGE, embed_to_vocab([description], time_steps, vocab)[0:1, :, :], i==0)

        print("SENTENCE:")
        gen_str = TEST_PREFIX
        for i in range(time_steps):
            element = np.random.choice( range(len(vocab)), p=out )
            gen_str += vocab[element]
            out = net.run_step( embed_to_vocab_2D(vocab[element], vocab), TEST_IMAGE, embed_to_vocab([description], time_steps, vocab)[0:1, :, :], False )
        print(gen_str)
    print()



lstm_size = 368 #128
num_layers = 2
batch_size = 500
time_steps = 75


ckpt_file = "/home/willie/workspace/TensorFlowWorkshop/data/SavedNetworks/model3"+str(1950)+"descriptions_mem.ckpt"
    
roasts, descriptions, images, valRoasts, valDescriptions, valImages=loadData('/home/willie/workspace/TensorFlowWorkshop/data/RedditRoasts/')

vocab = list(set((' '.join(roasts)).lower())|set((' '.join(descriptions)).lower())|set('~'))

roasts_embedded = embed_to_vocab(roasts, time_steps, vocab)
descriptions_embedded = embed_to_vocab(descriptions, time_steps, vocab)
valRoasts_embedded = embed_to_vocab(valRoasts, time_steps, vocab)
valDescriptions_embedded = embed_to_vocab(valDescriptions, time_steps, vocab)
in_size = out_size = len(vocab)


NUM_EPOCHS=500

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)

net = ModelNetwork(in_size = in_size,
                    lstm_size = lstm_size,
                    num_layers = num_layers,
                    out_size = out_size,
                    nnImageWidth=nnImageWidth,
                    nnImageHeight=nnImageHeight,
                    time_steps=time_steps,
                    session = sess,
                    learning_rate = 0.00003,
                    name = "char_rnn_network")

sess.run(tf.initialize_all_variables())

saver = tf.train.Saver(tf.all_variables())



## 1) TRAIN THE NETWORK
if ckpt_file == "/home/willie/workspace/TensorFlowWorkshop/data/SavedNetworks/model3"+str(1950)+"descriptions_mem.ckpt":
    last_time = time.time()
    saver.restore(sess, ckpt_file)
    

    for epochNum in range(0, NUM_EPOCHS):
        shuffle=np.random.permutation(len(roasts_embedded))
        for batchNum in range(0, len(roasts_embedded), batch_size):
            batchSize=min(batch_size, len(roasts_embedded)-batchNum)
            batch = roasts_embedded[shuffle[batchNum:batchNum+batchSize], :, :]
            batch[:,0,:]=embed_to_vocab_2D('~', vocab)
            batchImages=images[shuffle[batchNum:batchNum+batchSize], :, :, :]
            batchDescriptions=descriptions_embedded[shuffle[batchNum:batchNum+batchSize], :, :]
            batch_y = np.zeros((batchSize, time_steps, in_size))
            batch_y[:, :time_steps-1, :]=roasts_embedded[shuffle[batchNum:batchNum+batchSize], 1:, :]

            
            cst = net.train_batch(batch, batchImages, batchDescriptions, batch_y)
    
            if (batchNum%100) == 0:
                new_time = time.time()
                diff = new_time - last_time
                last_time = new_time
    
                print("epoch",epochNum,"batch: ",batchNum,"   loss: ",cst,"   speed: ",(100.0/diff)," batches / s")
        if epochNum%50==0:
            saver.save(sess, "/home/willie/workspace/TensorFlowWorkshop/data/SavedNetworks/model3"+str(epochNum)+"descriptions_mem.ckpt")






if ckpt_file != "":
    saver.restore(sess, ckpt_file)
    
getSentances(net, '/home/willie/Pictures/15134794_1663152383710584_93493203883920384_n.jpg', '22 year old computer science student roast me', time_steps, vocab)
getSentances(net, '/home/willie/Pictures/williamagnew.jpg', 'too hot to roast', time_steps, vocab)
getSentances(net, '/home/willie/Pictures/michael-kors.jpg', 'roast this daddy', time_steps, vocab)
getSentances(net, '/home/willie/Pictures/AlexLicc.jpg', 'he thinks he is a worm. roast him.', time_steps, vocab)
    
    