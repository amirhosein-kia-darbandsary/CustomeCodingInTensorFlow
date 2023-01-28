from tensorflow.keras.loasses import Loss
import tensorflow as tf
from typing import Optional


class HuberLoss(Loss):

    #initialize parameters
    def __init__(self , thereshold:Optional[int] = 1 ) -> None:
        super().__init__()
        self.thereshold = thereshold
    


    def __call__(self , y_true , y_pred) -> bool:
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= self.threshold
        small_error_loss = tf.square(error) / 2
        big_error_loss = self.threshold * (tf.abs(error) - (0.5 * self.threshold))
        return tf.where(is_small_error, small_error_loss, big_error_loss)