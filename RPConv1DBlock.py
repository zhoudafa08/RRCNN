import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
import tensorflow as tf
from tensorflow.python.keras.layers import Layer, InputSpec
#import keras
#from keras.engine.topology import Layer
#from keras.engine import InputSpec
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.regularizers import l2
from ReflectionPadding1D import ReflectionPadding1D
from tensorflow.keras.layers import Input

class RPConv1DBlock(Layer):
    def __init__(self, half_kernel_size=7, **kwargs):
        self.half_kernel_size = half_kernel_size
        self.RP = ReflectionPadding1D(padding=(self.half_kernel_size, 
                      self.half_kernel_size))
        #self.C1D = Conv1D(1, 2*self.half_kernel_size+1, use_bias=True, 
                      #kernel_regularizer=l2(1e-4))
        self.C1D_act = Conv1D(1, 2*self.half_kernel_size+1, 
                      activation='tanh', 
                      use_bias=True,
                      kernel_regularizer=l2(1e-4)
                      )
        super(RPConv1DBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel1 = self.add_weight(name='kernel1',
                        shape=(2*self.half_kernel_size+1, 1, 1),
                        initializer='uniform',
                        #regularizer=keras.regularizers.l1(1e-4),
                        trainable=True)
        #self.kernel2 = self.add_weight(name='kernel2',
        #                shape=(2*self.half_kernel_size+1, 1, 1),
        #                initializer='uniform',
        #                #regularizer=keras.regularizers.l1_l2(1e-4),
        #                trainable=True)
        super(RPConv1DBlock, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        shape = (input_shape[0],
              input_shape[1],
              input_shape[2]
              )
        return shape
    
    def call(self, inputs):
        x = self.RP(inputs)
        x = self.C1D_act(x)
        x = self.RP(x)
        filter1 = self.kernel1 ** 2
        filter1 = tf.scalar_mul(tf.math.reciprocal(tf.reduce_sum(filter1)), filter1)
        x = tf.nn.conv1d(x, filter1, stride=1, padding='VALID')
        #x = self.C1D(x)
        x = tf.keras.layers.Subtract()([inputs, x])
        return x
    
    def get_config(self):
        return {"half_kernel_size":self.half_kernel_size}
