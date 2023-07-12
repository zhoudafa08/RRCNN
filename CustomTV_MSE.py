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
#import keras

class CustomTV_MSE(tf.keras.losses.Loss):
    def __init__(self, regularization_factor=0.1, **kwargs):
        super().__init__(**kwargs)
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        error = tf.subtract(y_true, y_pred)
        mse = tf.math.reduce_mean(tf.square(error))
        # average
        #tv_error1 = tf.math.reduce_mean(tf.abs(error[:,:-1,:] - error[:,1:,:]))
        #tv_error2 = tf.math.reduce_mean(tf.abs(error[:,:-2,:] -2*error[:,1:-1,:] + error[:,2:,:]))
        #tv_error3 = tf.math.reduce_mean(tf.abs(error[:,:-3,:] - 3*error[:,1:-2,:] + 3*error[:,2:-1,:] - error[:,3:,:]))
        # noise
        #tv_pred1 = tf.math.reduce_mean(tf.abs(y_pred[:,:-1,:] - y_pred[:,1:,:]))
        #tv_pred2 = tf.math.reduce_mean(tf.abs(y_pred[:,:-2,:] - 2*y_pred[:,1:-1,:] + y_pred[:,2:,:]))
        #tv_pred3 = tf.math.reduce_mean(tf.abs(y_pred[:,:-3,:] - 3*y_pred[:,1:-2,:] + 3*y_pred[:,2:-1,:] - y_pred[:,3:,:]))

        # signal decomposition
        length = y_pred.shape[1]
        ### denoising
        ##noise_error = error[:int(length/3)]
        ##cond = noise_error < 0
        ##noise_loss = tf.reduce_mean((1-0.3)*noise_error[cond]**2) + tf.reduce_mean(0.3*noise_error[~cond]**2)

        y_pred1 = y_pred[:, :int(length/2), :]
        y_pred2 = y_pred[:, int(length/2):, :]
        tv_pred11 = tf.math.reduce_mean(tf.abs(y_pred1[:,:-1,:] - y_pred1[:,1:,:]))
        tv_pred12 = tf.math.reduce_mean(tf.abs(y_pred1[:,:-2,:] - 2*y_pred1[:,1:-1,:] + y_pred1[:,2:,:]))
        tv_pred13 = tf.math.reduce_mean(tf.abs(y_pred1[:,:-3,:] - 3*y_pred1[:,1:-2,:] + 3*y_pred1[:,2:-1,:] - y_pred1[:,3:,:]))
        tv_pred21 = tf.math.reduce_mean(tf.abs(y_pred2[:,:-1,:] - y_pred2[:,1:,:]))
        tv_pred22 = tf.math.reduce_mean(tf.abs(y_pred2[:,:-2,:] - 2*y_pred2[:,1:-1,:] + y_pred2[:,2:,:]))
        tv_pred23 = tf.math.reduce_mean(tf.abs(y_pred2[:,:-3,:] - 3*y_pred2[:,1:-2,:] + 3*y_pred2[:,2:-1,:] - y_pred2[:,3:,:]))
        
        #orthogonal
        orthogonal = tf.math.divide(tf.square(tf.multiply(y_pred1, y_pred2)), tf.norm(y_pred1)*tf.norm(y_pred2))
        #return mse + (2*tv_error1 + 3*tv_error2 + 4*tv_error3) * self.regularization_factor
        #return mse + (3*tv_pred1 + 4*tv_pred2 + 5*tv_pred3) * self.regularization_factor
        #return mse + orthogonal + (5*tv_pred11 + 5*tv_pred13 + 4*tv_pred21 + 4*tv_pred23)*self.regularization_factor
        return mse +  (5*tv_pred11 + tv_pred13 + tv_pred21 + tv_pred23)*self.regularization_factor
        #return mse + orthogonal
        #return mse + orthogonal + (3*tv_pred11 + 3*tv_pred12 + 3*tv_pred13 + 2*tv_pred21 + 2*tv_pred22 + 2*tv_pred23)*self.regularization_factor
        #return mse + (5*tv_pred11 + 4*tv_pred12 + 5*tv_pred13+2*tv_pred21 + 2*tv_pred23) * self.regularization_factor
    
