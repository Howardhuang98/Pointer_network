#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   test.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/11/9 15:31  
------------      
"""
from tensorflow import int32

from TSP_data import Tsp
from model import *
import tensorflow as tf
from tensorflow.keras.layers import LSTMCell


p = Tsp()
x,y = p.next_batch(5)
x = x.astype('float64')
x = tf.constant(x)
print(x.shape)
en = Encoder()
de = Decoder()
enc_output, state_h, state_c = en(x)
y = de(enc_output, [state_h, state_c])
print(y.shape)
