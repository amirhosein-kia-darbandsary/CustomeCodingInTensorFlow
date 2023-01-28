"""
in here we want add activation function to our custom layer so we need to recieve what kind of activation
from parameters then use it 
"""

import tensorflow as tf 
from tensorflow.keras.layers import Layer
from typing import Optional

class SimpleDense(Layer):
    
    def __init__(self , units: Optional[int]=32 , activation: Optional[str] = None):
        super(SimpleDense, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

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
        return self.activation(tf.matmul(inputs, self.w) + self.b)

# How to use this ?

# use the Sequential API to build a model with our custom layer
my_layer = SimpleDense(units=1)
model = tf.keras.Sequential([my_layer(32 , 'relu')])
