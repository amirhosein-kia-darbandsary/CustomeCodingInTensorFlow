"""
we want to build a custom model in class model shape it has many benefits that you can use
lkie have a loop on your layers to don't repeat a layer for  many times.

in constructor we need to define the layers then we need to define their functionality in call
function 
like the example we define the out put layers and hidden layers in __init__
then we defines the network architecture in __call__ method 

"""

from tensorflow.keras.layers import Input , Dense , concatenate , Model
from typing import Optional

class CustomModel(Model):
    
    def __init__(self, units:Optional[int]=30 , activation:Optional[str]='relu' , **kwargs ):
        super().__init__(**kwargs)
        self.layer1 = Dense(units=units, activation=activation)
        self.layer2 = Dense(units=units, activation=activation)
        self.aux_output = Dense(1)
        self.main_output = Dense(1)

    
    def __call__(self , input):

        input_a , input_b = input
        layer1 = layer1(input_a)
        layer2 = layer2(self.layer1)
        concat_layer = concatenate(self.layer2 , input_b)
        aux_output = aux_output(self.layer2)
        main_output = main_output(concat_layer)
        
        return main_output , aux_output

# how to use ? 
model = CustomModel()

