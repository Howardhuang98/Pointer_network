#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   test_decoder.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/11/15 11:23  
------------      
"""
from model import *

x = tf.random.normal(shape=(5, 10, 2))
encoder_outputs = tf.random.normal(shape=(5, 10, 128))
state = [tf.random.normal(shape=(5, 128)), tf.random.normal(shape=(5, 128))]
dec = Decoder()
y = dec(x, encoder_outputs, state)
print(y[0])
