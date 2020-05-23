import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
import os
import wave
import tensorflow.keras
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, GRU, Bidirectional, Conv1D, Attention, Add, BatchNormalization, Multiply, Activation
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta, SGD, Adam, RMSprop
from sklearn.metrics import f1_score, log_loss, accuracy_score
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.models import load_model
import gc
import random
import tensorflow as tf
import wave
import warnings
import sys
warnings.simplefilter('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.random.set_seed(SEED)

n_train_data = 720
n_classes = 4
min_seconds = 40000
max_seconds = 140000
total_seconds = max_seconds - min_seconds
median_seconds = min_seconds + total_seconds
print('total seconds: {}ms'.format(total_seconds))

class WCF():
    def __init__(self):
        self.model = load_model('./best_wavenet_model_1.h5')

    def to_feeling(self, y_pred):
        if y_pred == 0:
            return 'good'
        elif y_pred == 1:
            return 'sad'
        elif y_pred == 2:
            return 'angry'
        else:
            return 'surprise'

    def preprocess(self, data_path):
        wave_file = wave.open(data_path,"r")
        x = wave_file.readframes(wave_file.getnframes())
        x = np.frombuffer(x, dtype= "int16")
        if x.shape[0] < total_seconds:
            n_pad = total_seconds - x.shape[0]
            x = np.pad(x, [min_seconds, n_pad], 'mean')
        if x.shape[0] < max_seconds:
            n_pad = max_seconds - x.shape[0]
            x = np.pad(x, [0, n_pad], 'mean')

        x = x[min_seconds: max_seconds]
        wave_file.close()
        mean = x.mean()
        std = x.std()
        X = ((x - mean) / std).reshape(1, -1, 1)
        return X

    def predict(self, data_path):
        X = self.preprocess(data_path)
        y_pred = self.model.predict(X.reshape(1, total_seconds, 1)).argmax(1)
        feeling = self.to_feeling(y_pred)
        return feeling

model = WCF()
path = './output.wav'
y = model.predict(path)
print(y)