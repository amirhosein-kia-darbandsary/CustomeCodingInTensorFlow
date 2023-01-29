"""
As shown in the lectures, we will first implement the Identity Block which contains 
the skip connections (i.e. the add() operation below
This will also inherit the Model class and implement the __init__() and call() methods.
"""
import tensorflow as tf
from tensorflow.keras.models import Model


class IdentityBlock(Model):

    def __init__(self , kernel_size , filters ) -> None:
        # identify layers
        super(IdentityBlock).__init__()


        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.relu = tf.keras.Activation('relu')
        self.add = tf.keras.layers.Add()

    def __call__(self , input ):
        
        layer = self.conv1(input)
        layer = self.bn1(layer)
        
        layer = self.relu(layer)

        layer = self.conv2(layer)
        layer = self.bn2(layer)
        layer = self.add(input , layer)
        return self.relu(layer) 

        

class ResNet(tf.keras.Model):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.conv = tf.keras.layers.Conv2D(64, 7, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')
        self.max_pool = tf.keras.layers.MaxPool2D((3, 3))

        # Use the Identity blocks that you just defined
        self.id1a = IdentityBlock(64, 3)
        self.id1b = IdentityBlock(64, 3)

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        x = self.max_pool(x)

        # insert the identity blocks in the middle of the network
        x = self.id1a(x)
        x = self.id1b(x)

        x = self.global_pool(x)
        return self.classifier(x)