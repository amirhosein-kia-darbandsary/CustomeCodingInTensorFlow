"""
For building loss function in Fuction shape we need 2 parameters 
y_true : the real data we have.  
y_prdict : the pridicted answer with model

For Example we implement Huber loss function:
"""

import tensorflow as tf

def huber_loss_function(y_true, y_pred):
    threshold = 1
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= threshold
    small_error_loss = tf.square(error) / 2
    big_error_loss = threshold * (tf.abs(error) - (0.5 * threshold))
    return tf.where(is_small_error, small_error_loss, big_error_loss)

"""
How to use inside our code ? 
like the above code : 

model.compile(optimizer = "Optimizer you want to use " , loss = huber_loss_function)
"""



"""
if we need add extra parameter to loss fucntion we need to have wrapper function like the blow :
"""

def huber_loss_function(threshold):
    def loss_function(y_true  , y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= threshold
        small_error_loss = tf.square(error) / 2
        big_error_loss = threshold * (tf.abs(error) - (0.5 * threshold))
        return tf.where(is_small_error, small_error_loss, big_error_loss)
    return loss_function