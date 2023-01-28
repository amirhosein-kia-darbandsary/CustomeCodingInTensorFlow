"""
the first way is build a simple layer with Lambda layer in tensorflow like the example : 

"""

from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Sequencials
import tensorflow as tf 


# in here we just make a simple layer that power the input  :  Lambda(x : x**2)

# how to use ?

model = Sequencial([
    tf.keras.layers.Dense(input_shape=(20,20) , activation = 'relu') , 
    Lambda(x : x**2),
    tf.keras.layers.Dense(activation = 'softmax')
]) 


"""
the secoond way is build a external function then pass it to the lambda layer lkie the example : 

"""
def custom_relu(input):
    return tf.maximum(0.0 , input)

layer = tf.keras.layers.Lambda(custom_relu(input))



"""
the third way  : 
Custom Layer with weights
To make custom layer that is trainable,
we need to define a class that inherits the Layer base class from Keras. 
This class requires three functions: __init__(), build() and call().
These ensure that our custom layer has a state and computation 
that can be accessed during training or inference.

"""
from tensorflow.keras.layers import Layer
from typing import Optional
class SimpleDense(Layer):
    
    def __init__(self , units: Optional[int]=32):
        super(SimpleDense, self).__init__()
        self.units = units
    

    def __build__(self , input_shape):
        """Create the state of the layer (weights)"""

        # initialize new weights 

        w_init = w_init = tf.random_normal_initializer()
        self.w = tf.Variable(name = 'kernel' ,
                            initial_value = w_init(shape=(input_shape[-1], self.units),
                                 dtype='float32'),
                                 trainable = True)



        # initialize the biases
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(name="bias",
            initial_value=b_init(shape=(self.units,), dtype='float32'),
            trainable=True)


        
    def call(self, inputs):
        '''Defines the computation from inputs to outputs'''
        return tf.matmul(inputs, self.w) + self.b

# How to use this ?

# use the Sequential API to build a model with our custom layer
my_layer = SimpleDense(units=1)
model = tf.keras.Sequential([my_layer])
