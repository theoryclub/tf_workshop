import tensorflow as tf #import TensorFlow library, same as Java import

hello=tf.constant('Hello World!') #Create a variable in the tensorflow graph
session=tf.Session()    #Compile the tensorflow graph
print(session.run(hello)) #print the value of the variable hello