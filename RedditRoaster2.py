import tensorflow as tf
import numpy as np
import random
import time
import sys
import os

numSymbols=500
num_steps=10
numberInputs=3
numLayers=4
lstm_size=5

imageWidth=600
imageHeight=768

nnImageWidth=25
nnImageHeight=32


## RNN with num_layers LSTM layers and a fully-connected output layer
## The network allows for a dynamic number of iterations, depending on the inputs it receives.
##
##    out   (fc layer; out_size)
##     ^
##    lstm
##     ^
##    lstm  (lstm size)
##     ^
##     in   (in_size)
class ModelNetwork:
    
    
        
        
    #     descriptions=np.zeros([int(7*numRoasts/8), num_steps, numSymbols], np.float32)
    #     images=np.zeros([int(7*numRoasts/8), nnImageWidth, nnImageHeight, 3], np.float32)
    #     roasts=np.zeros([int(7*numRoasts/8), num_steps, numSymbols], np.float32)
    #       
    #     valDescriptions=np.zeros([int(math.ceil(numRoasts/8)), num_steps, numSymbols], np.float32)
    #     valImages=np.zeros([int(math.ceil(numRoasts/8)), nnImageWidth, nnImageHeight, 3], np.float32)
    #     valRoasts=np.zeros([int(math.ceil(numRoasts/8)), num_steps, numSymbols], np.float32)
        
        
#         descriptions=np.zeros([10, num_steps, numSymbols], np.float32)
#         images=np.zeros([10, nnImageWidth, nnImageHeight, 3], np.float32)
#         roasts=np.zeros([10, num_steps, numSymbols], np.float32)
#           
#         valDescriptions=np.zeros([10, num_steps, numSymbols], np.float32)
#         valImages=np.zeros([int(numRoasts/8), nnImageWidth, nnImageHeight, 3], np.float32)
#         valRoasts=np.zeros([int(numRoasts/8), num_steps, numSymbols], np.float32)
#         
#         
#         dataInd=0
#         for fileName in os.listdir(file):
#             if fileName.startswith('roasts'):
#                 
#                 roast=self.readToLines(file+fileName)
#                 for roastInd in range(0, min(3, len(roast))):
#                     roastWords=roast[roastInd].split(' ')
#                     for wordInd in range(0, min(num_steps, len(roastWords))):
#                         if wordMapping.has_key(roastWords[wordInd]):
#                             if dataInd<int(7*numRoasts/8):
#                                 roasts[dataInd, wordInd, :]=wordMapping[roastWords[wordInd]]
#                             else:
#                                 valRoasts[dataInd-int(7*numRoasts/8), wordInd, :]=wordMapping[roastWords[wordInd]]
#                         else:
#                             if dataInd<int(7*numRoasts/8):
#                                 roasts[dataInd, wordInd, numSymbols-1]=1.0
#                             else:
#                                 valRoasts[dataInd-int(7*numRoasts/8), wordInd, numSymbols-1]=1.0
#                 
#                     roastEndInd=fileName.index('roasts')+len('roasts')
#                     pageNumEndInd=fileName.index('_')
#                     pageNum=fileName[roastEndInd:pageNumEndInd]
#                     postNumEndInd=fileName.index('.')
#                     postNum=fileName[pageNumEndInd+1:postNumEndInd]
#                     
#                     imageFile=open(file+'image'+pageNum+'_'+postNum+'.jpeg', 'r+b')
#                     image=Image.open(imageFile)
#                     unscaledImages+=[image]
#                     image=resizeimage.resize_cover(image, [imageWidth, imageHeight], validate=False)
#                     image=image.resize((nnImageWidth, nnImageHeight),  PIL.Image.ANTIALIAS)
#                     #image.show()
#                     pixels=image.load()
#                     for x in range(0, nnImageWidth):
#                         for y in range(0, nnImageHeight):
#                             for rgbInd in range(0, 3):
#                                 if dataInd<int(7*numRoasts/8):
#                                     i=pixels[x, y]
#                                     images[dataInd, x, y, rgbInd]=pixels[x, y][rgbInd]/255.0
#                                 else:
#                                     valImages[dataInd-int(7*numRoasts/8), x, y, rgbInd]=pixels[x, y][rgbInd]/255.0
#                     
#                     description=self.readToLines(file+'desc'+pageNum+'_'+postNum+'.txt')[0]
#                     descriptionWords=description.split(' ')
#                     for wordInd in range(0, min(num_steps, len(descriptionWords))):
#                         if wordMapping.has_key(descriptionWords[wordInd]):
#                             if dataInd<int(7*numRoasts/8):
#                                 descriptions[dataInd, num_steps-1-wordInd]=wordMapping[descriptionWords[wordInd]]
#                             else:
#                                 valDescriptions[dataInd-int(7*numRoasts/8), num_steps-1-wordInd]=wordMapping[descriptionWords[wordInd]]       
#                     dataInd+=1
#                     imageFile.close()
#             
#                     if dataInd>=10:
#                         break;
#                 if dataInd>=10:
#                     break;
#         
#         return [descriptions, images, roasts, valDescriptions, valImages, valRoasts]
    
    
    def __init__(self, in_size, lstm_size, num_layers, out_size, session, learning_rate=0.003, name="rnn"):
        self.scope = name

        self.in_size = in_size
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.out_size = out_size

        self.session = session

        self.learning_rate = tf.constant( learning_rate )

        # Last state of LSTM, used when running the network in TEST mode
        self.lstm_last_state = np.zeros((self.num_layers*2*self.lstm_size,))

        with tf.variable_scope(self.scope):
            ## (batch_size, timesteps, in_size)
            self.xinput = tf.placeholder(tf.float32, shape=(None, None, self.in_size), name="xinput")
            self.lstm_init_value = tf.placeholder(tf.float32, shape=(None, self.num_layers*2*self.lstm_size), name="lstm_init_value")

            # LSTM
            self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size, forget_bias=1.0, state_is_tuple=False)
            self.lstm = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell] * self.num_layers, state_is_tuple=False)

            # Iteratively compute output of recurrent network
            outputs, self.lstm_new_state = tf.nn.dynamic_rnn(self.lstm, self.xinput, initial_state=self.lstm_init_value, dtype=tf.float32)

            # Linear activation (FC layer on top of the LSTM net)
            self.rnn_out_W = tf.Variable(tf.random_normal( (self.lstm_size, self.out_size), stddev=0.01 ))
            self.rnn_out_B = tf.Variable(tf.random_normal( (self.out_size, ), stddev=0.01 ))

            outputs_reshaped = tf.reshape( outputs, [-1, self.lstm_size] )
            network_output = ( tf.matmul( outputs_reshaped, self.rnn_out_W ) + self.rnn_out_B )

            batch_time_shape = tf.shape(outputs)
            self.final_outputs = tf.reshape( tf.nn.softmax( network_output), (batch_time_shape[0], batch_time_shape[1], self.out_size) )


            ## Training: provide target outputs for supervised training.
            self.y_batch = tf.placeholder(tf.float32, (None, None, self.out_size))
            y_batch_long = tf.reshape(self.y_batch, [-1, self.out_size])

            self.cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(network_output, y_batch_long) )
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate, 0.9).minimize(self.cost)


    ## Input: X is a single element, not a list!
    def run_step(self, x, init_zero_state=True):
        ## Reset the initial state of the network.
        if init_zero_state:
            init_value = np.zeros((self.num_layers*2*self.lstm_size,))
        else:
            init_value = self.lstm_last_state

        out, next_lstm_state = self.session.run([self.final_outputs, self.lstm_new_state], feed_dict={self.xinput:[x], self.lstm_init_value:[init_value]   } )

        self.lstm_last_state = next_lstm_state[0]

        return out[0][0]


    ## xbatch must be (batch_size, timesteps, input_size)
    ## ybatch must be (batch_size, timesteps, output_size)
    def train_batch(self, xbatch, ybatch):
        init_value = np.zeros((xbatch.shape[0], self.num_layers*2*self.lstm_size))

        cost, _ = self.session.run([self.cost, self.train_op], feed_dict={self.xinput:xbatch, self.y_batch:ybatch, self.lstm_init_value:init_value   } )

        return cost




sentances=''
    
def readToLines(file):
    csvFile=open(file)
    lines=csvFile.read().splitlines()
    return lines

def loadData(file):
    numRoasts=0.0
    wordDict={}
    global sentances
    for fileName in os.listdir(file):
        if fileName.startswith('roasts'):
            roasts=readToLines(file+fileName)
            for roastInd in range(0, min(3, len(roasts))):
                sentances+=roasts[roastInd]+' '
                roastWords=roasts[roastInd].split(' ')
                for word in roastWords:
                    if not wordDict.has_key(word):
                        wordDict[word]=0
                    wordDict[word]+=1   
            numRoasts+=min(3, len(roasts))
        elif fileName.startswith('desc'):
            desc=readToLines(file+fileName)
            sentances+=desc[0]+' '
            word=desc[0].split(' ')
            #sentances+=[word]
            for word in roastWords:
                if not wordDict.has_key(word):
                    wordDict[word]=0
                wordDict[word]+=1
                

# Embed string to character-arrays -- it generates an array len(data) x len(vocab)
# Vocab is a list of elements
def embed_to_vocab(data_, vocab):
    data = np.zeros((len(data_), len(vocab)))

    cnt=0
    for s in data_:
        v = [0.0]*len(vocab)
        v[vocab.index(s)] = 1.0
        data[cnt, :] = v
        cnt += 1

    return data

def decode_embed(array, vocab):
    return vocab[ array.index(1) ]






ckpt_file = ""
TEST_PREFIX = "you look " # Prefix to prompt the network in test mode

print("Usage:")
print('\t\t ', sys.argv[0], ' [ckpt model to load] [prefix, e.g., "The "]')
if len(sys.argv)>=2:
    ckpt_file=sys.argv[1]
if len(sys.argv)==3:
    TEST_PREFIX = sys.argv[2]




## Load the data
# data_ = ""
# with open('data/shakespeare.txt', 'r') as f:
#     data_ += f.read()
    
loadData('/home/willie/workspace/TensorFlowWorkshop/data/RedditRoasts/')
data_=sentances
data_ = data_.lower()

## Convert to 1-hot coding
vocab = list(set(data_))

data = embed_to_vocab(data_, vocab)


in_size = out_size = len(vocab)
lstm_size = 256 #128
num_layers = 2
batch_size = 64 #128
time_steps = 100 #50

NUM_TRAIN_BATCHES = 20000

LEN_TEST_TEXT = 500 # Number of test characters of text to generate after training the network



## Initialize the network
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)

net = ModelNetwork(in_size = in_size,
                    lstm_size = lstm_size,
                    num_layers = num_layers,
                    out_size = out_size,
                    session = sess,
                    learning_rate = 0.003,
                    name = "char_rnn_network")

sess.run(tf.initialize_all_variables())

saver = tf.train.Saver(tf.all_variables())



## 1) TRAIN THE NETWORK
if ckpt_file == "":
    last_time = time.time()

    batch = np.zeros((batch_size, time_steps, in_size))
    batch_y = np.zeros((batch_size, time_steps, in_size))

    possible_batch_ids = range(data.shape[0]-time_steps-1)
    for i in range(NUM_TRAIN_BATCHES):
        # Sample time_steps consecutive samples from the dataset text file
        batch_id = random.sample( possible_batch_ids, batch_size )

        for j in range(time_steps):
            ind1 = [k+j for k in batch_id]
            ind2 = [k+j+1 for k in batch_id]

            batch[:, j, :] = data[ind1, :]
            batch_y[:, j, :] = data[ind2, :]


        cst = net.train_batch(batch, batch_y)

        if (i%100) == 0:
            new_time = time.time()
            diff = new_time - last_time
            last_time = new_time

            print("batch: ",i,"   loss: ",cst,"   speed: ",(100.0/diff)," batches / s")

    saver.save(sess, "/home/willie/workspace/TensorFlowWorkshop/data/SavedNetworks/model.ckpt")




## 2) GENERATE LEN_TEST_TEXT CHARACTERS USING THE TRAINED NETWORK

if ckpt_file != "":
    saver.restore(sess, ckpt_file)

TEST_PREFIX = TEST_PREFIX.lower()
for i in range(len(TEST_PREFIX)):
    out = net.run_step( embed_to_vocab(TEST_PREFIX[i], vocab) , i==0)

print("SENTENCE:")
gen_str = TEST_PREFIX
for i in range(LEN_TEST_TEXT):
    element = np.random.choice( range(len(vocab)), p=out ) # Sample character from the network according to the generated output probabilities
    gen_str += vocab[element]

    out = net.run_step( embed_to_vocab(vocab[element], vocab) , False )
print(gen_str)