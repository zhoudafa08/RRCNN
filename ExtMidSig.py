import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
import tensorflow as tf
from tensorflow.python.keras.layers import Layer, InputSpec
#from keras.engine.topology import Layer
#from keras.engine import InputSpec

class ExtMidSig(Layer):
    def __init__(self, x_length, y_length, **kwargs):
        self.x_length = x_length
        self.y_length = y_length
        self.exd_length = int((self.x_length-self.y_length)/2)
        super(ExtMidSig, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        shape = (input_shape[0],
              intput_shape[1]-2*self.exd_length,
              input_shape[2]
              )
        return shape

    def call(self, input_tensor, mask=None):
        return input_tensor[:, self.exd_length : -self.exd_length, :] 
    
    def get_config(self):
        return {"x_length":self.x_length, "y_length":self.y_length} 
