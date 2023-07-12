from tensorflow.keras.models import Model, load_model, model_from_json
from tensorflow.keras.layers import Conv1D, Input, Activation
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.python.keras.layers import Layer, InputSpec
from ExtMidSig import ExtMidSig
from CustomTV_MSE import CustomTV_MSE
from ReflectionPadding1D import ReflectionPadding1D
from RPConv1DBlock import RPConv1DBlock
import numpy as np
import pandas as pd
import sys,pywt
from sklearn.metrics import mean_squared_error
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
import scipy.signal as ss
import time

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# Generate data
def data():
    length = 2400
    exd_length = int(0.5*length)
    mode = 'symmetric'
    x_train = np.empty([length+2*exd_length, 1])
    y_train = np.empty([2*length, 1])
    t = np.linspace(0, 6, length)
    
    for j in range(5, 15):
        x1 = np.cos(j *np.pi * t)
        x2 = np.cos((j+1.5)*np.pi*t)
        tmp_x = pywt.pad(x1 + x2, exd_length, mode)
        x_train = np.c_[x_train, tmp_x]
        y_train = np.c_[y_train, np.r_[x2, x1]]
        tmp_x = pywt.pad(x2, exd_length, mode)
        x_train = np.c_[x_train, tmp_x]
        y_train = np.c_[y_train, np.r_[x2, 0.0*t]]
        x2 = np.cos((j+1.5)*np.pi*t + t * t + np.cos(t))
        tmp_x = pywt.pad(x1 + x2, exd_length, mode)
        x_train = np.c_[x_train, tmp_x]
        y_train = np.c_[y_train, np.r_[x2, x1]]
        tmp_x = pywt.pad(x2, exd_length, mode)
        x_train = np.c_[x_train, tmp_x]
        y_train = np.c_[y_train, np.r_[x2, 0.0*t]]
        for i in range(2, 20):
            x2 = np.cos(j*i*np.pi*t)
            tmp_x = pywt.pad(x1 + x2, exd_length, mode)
            x_train = np.c_[x_train, tmp_x]
            y_train = np.c_[y_train, np.r_[x2, x1]]
            tmp_x = pywt.pad(x2, exd_length, mode)
            x_train = np.c_[x_train, tmp_x]
            y_train = np.c_[y_train, np.r_[x2, 0.0*t]]
            x2 = np.cos(j*i*np.pi*t + t * t + np.cos(t))
            tmp_x = pywt.pad(x1 + x2, exd_length, mode)
            x_train = np.c_[x_train, tmp_x]
            y_train = np.c_[y_train, np.r_[x2, x1]]
            tmp_x = pywt.pad(x2, exd_length, mode)
            x_train = np.c_[x_train, tmp_x]
            y_train = np.c_[y_train, np.r_[x2, 0.0*t]]
    
    x1 = 0.0 * t
    for i in range(1, 15):
        x2 = 0.1 * i * t
        tmp_x = pywt.pad(x1 + x2, exd_length, mode)
        x_train = np.c_[x_train, tmp_x]
        y_train = np.c_[y_train, np.r_[x1, x2]]
        tmp_x = pywt.pad(x1 + 0.1 * i, exd_length, mode)
        x_train = np.c_[x_train, tmp_x]
        y_train = np.c_[y_train, np.r_[x1, x1+0.1*i]]
    
    x_train = np.delete(x_train, 0, axis=1)
    y_train = np.delete(y_train, 0, axis=1)
    indices = np.arange(x_train.shape[1])
    np.random.seed(0)
    np.random.shuffle(indices)
    x_sample = x_train[:, indices]
    y_sample = y_train[:, indices]
    train_num = int(0.7*x_sample.shape[1])
    x_train = x_sample[:, :train_num]
    x_test = x_sample[:, train_num:]
    y_train = y_sample[:, :train_num]
    y_test = y_sample[:, train_num:]
    
    x_train = x_train.transpose().reshape(-1, length+2*exd_length, 1)
    y_train = y_train.transpose().reshape(-1, 2*length, 1)
    x_test = x_test.transpose().reshape(-1, length+2*exd_length, 1)
    y_test = y_test.transpose().reshape(-1, 2*length, 1)
    print(x_train.shape, y_train.shape)
    return x_train, y_train, x_test, y_test

# construct model
def create_model(x_train, y_train, x_test, y_test):
    #Layer_num = {{choice(range(5, 16))}}
    Layer_num = 12
    inputs = Input(shape=x_train.shape[1:], dtype='float32')
    ## cell 1
    outs = RPConv1DBlock({{choice(range(20, 100, 5))}})(inputs)
    outs = RPConv1DBlock({{choice(range(20, 100, 5))}})(outs)
    outs = RPConv1DBlock({{choice(range(20, 100, 5))}})(outs)
    outs = RPConv1DBlock({{choice(range(20, 100, 5))}})(outs)
    outs = RPConv1DBlock({{choice(range(20, 100, 5))}})(outs)
    outs = RPConv1DBlock({{choice(range(20, 100, 5))}})(outs)
    outs = RPConv1DBlock({{choice(range(20, 100, 5))}})(outs)
    outs = RPConv1DBlock({{choice(range(20, 100, 5))}})(outs)
    for subnet in range(8, Layer_num):
        outs = RPConv1DBlock(int({{choice(range(20, 100, 5))}}/np.sqrt(1)))(outs)
    outs1 = ExtMidSig(x_train.shape[1], int(y_train.shape[1]/2))(outs)
    
    ## cell 1
    outs = inputs - outs
    outs = RPConv1DBlock({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock({{choice(range(10, 150, 5))}})(outs)
    outs = RPConv1DBlock({{choice(range(10, 150, 5))}})(outs)
    for subnet in range(8, Layer_num):
        outs = RPConv1DBlock(int({{choice(range(10, 150, 5))}}/np.sqrt(1)))(outs)
    outs2 = ExtMidSig(x_train.shape[1], int(y_train.shape[1]/2))(outs)
    
    final_outs = tf.keras.layers.Concatenate(axis=1)([outs1, outs2]) 
    model = Model(inputs=inputs, outputs=final_outs)
    model.summary()
    adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer='adam', loss=CustomTV_MSE(0.05), metrics=['mse'])
     
    # train model
    model.fit(x=x_train, y=y_train, epochs=25, 
                batch_size=16, verbose=1, validation_data=(x_test, y_test))
    
    pred = model.predict(x_test)
    score = mean_squared_error(pred[:,:,0], y_test[:, :,0])
    weight_path = './models/model_rrcnn.h5'
    try:
        min_error = min_error
    except UnboundLocalError:
        min_error = 10
    if score <= min_error:
        model.save(weight_path)
        min_error = score
    sys.stdout.flush()
    return {'loss': score, 'model': model, 'status': STATUS_OK}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                            data=data, algo=tpe.suggest,
                            max_evals=5, trials=Trials())
    
    best_model = load_model('./models/model_rrcnn.h5', \
            custom_objects={'ReflectionPadding1D':ReflectionPadding1D, 'RPConv1DBlock':RPConv1DBlock, 'ExtMidSig':ExtMidSig, 'CustomTV_MSE':CustomTV_MSE}) 
    
    length = 2400
    exd_length = int(0.5*length)
    t = np.linspace(0, 6, length)
    x1 = np.cos(5*np.pi*t)
    x2 = np.cos(7*np.pi*t+t*t+np.cos(t))
    signal = pywt.pad(x1+x2, exd_length, 'symmetric')
    signal = signal.reshape(-1,1)
    layer_model = Model(inputs=best_model.input, outputs=best_model.layers[-1].output)
    start = time.time()
    pred = layer_model.predict(signal.reshape(1,length+2*exd_length,1))
    end = time.time()
    print('Time:', end-start)
